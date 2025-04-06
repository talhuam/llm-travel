"""
BuDing大模型的模型架构，
模型架构参考tinyllm(模型参考llama2)、qwen2、llama
tinyllm: https://github.com/wdndev/tiny-llm-zh/blob/main/train/modeling_tinyllm.py
qwen2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
llama: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
import math
from typing import Optional, Tuple, Union, List
import warnings
import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from configuration_buding import BuDingConfig

logger = logging.get_logger(__name__)


class BuDingMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class BuDingRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BuDingRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class BuDingRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        旋转位置编码
        - dim (int): 旋转嵌入的维度大小。
        - max_position_embeddings (int): 预计算的最大位置嵌入数，默认为2048。
        - base (int): 用于计算逆频率的基本频率，默认为10000。\theta=10000 ** (-2i/d)
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算逆频率值，并将其注册为模型的缓冲区，
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了支持`torch.jit.trace`功能，立即计算预存储的余弦和正弦缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        预计算的余弦和正弦缓存
        """
        self.max_seq_len_cached = seq_len
        # 创建一个从0到最大序列长度-1的整数张量，与 inv_freq 具有相同的设备和数据类型
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算每个位置与每个维度的频率，形成频谱矩阵，shape为: [seq_len, inv_freq]，计算m\theta
        freqs = torch.outer(t, self.inv_freq)

        # 不同于论文中的实现，这里采用了不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x shape: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """
    旋转输入一半的 hidden dim
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    应用旋转位置编码
    :param q: query向量
    :param k: key向量
    :param cos: cos(m * \theta)
    :param sin: sin(m * \theta)
    :param position_ids: 与q和k对应位置的标记索引。例如，在处理KV缓存时，可以使用偏移过的位置ID。shape: [batch_size, seq_len]
    :param unsqueeze_dim: 维度扩展，方便广播
    :return: 应用旋转位置编码后的query和key
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class BuDingAttention(nn.Module):
    """
    多头注意力机制，采用GQA，减少kv cache
    """
    def __init__(self, config: BuDingConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        # 因果自回归模式
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = BuDingRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # 转化成多头并行结构
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # query和key应用旋转位置编码，传值value_states只为获取张量类型，不会对value_states进行旋转
        ## 1、获取cos(m\theta) 和 sin(m\theta), shape: [seq_len, 2 * inv_freq] = [seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        ## 2、应用旋转位置编码
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 如果存在缓存，则更新 kv
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 如果 num_key_value_heads 小于 num_heads，则重复key和value向量以匹配头数量
        ## 1、如果num_key_value_heads = num_heads,则是MHA
        ## 2、如果num_key_value_heads = 1,则是MQA,kv cache是MHA的1/num_heads
        ## 3、如果如果num_key_value_heads = num_heads/num_key_value_groups,则是GQA,kv cache是MHA的groups/num_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力权重
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # softmax归一化注意力权重，并转换至float32类型以防止数值溢出
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # 注意力输出
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 还原注意力输出的形状以与后续层对接
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 通过o_proj层进一步处理注意力输出
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BuDingSDPAAttention(BuDingAttention):
    """
    使用 torch.nn.functional.scaled_dot_product_attention 实现的注意力模块。
    该模块继承自 `BuDingAttention`，因为模块的权重保持不变。唯一的变化在于前向传播过程中适应 SDPA API。
    Scaled Dot Product Attention (SDPA)
    """

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 当设置output_attentions=True时，由于torch.nn.functional.scaled_dot_product_attention不支持直接返回注意力权重
        # 因此暂时降级回用父类的手动实现方式，并发出警告提示用户未来版本的更改要求
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Model is using SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        # 获取输入维度信息
        bsz, q_len, _ = hidden_states.size()

        # 对输入进行线性映射得到query、key、value向量
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 将映射后的向量调整为多头注意力所需格式
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 计算有效的 kv 序列长度（考虑缓存的情况）
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # 应用旋转位置嵌入（RoPE）
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 如果有缓存，更新key和value状态
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # 使用scaled_dot_product_attention进行计算
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        # 还原注意力输出的形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # 将注意力输出通过最终的线性层（o_proj层）
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


BUDING_ATTENTION_CLASSES = {
    "eager": BuDingAttention,
    "sdpa": BuDingSDPAAttention
}


class BuDingDecoderLayer(nn.Module):
    def __init__(self, config: BuDingConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = BUDING_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = BuDingMLP(config)
        self.input_layernorm = BuDingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BuDingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入形状 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask 形状`(batch, sequence_length)`，
                填充使用0表示
            position_ids: shape [batch_size, seq_len]
            output_attentions (`bool`, *optional*): 是否返回所有注意力层的注意力张量。
            use_cache (`bool`, *optional*): 如果设置为 `True`，则返回 `past_key_values` 关键值状态，可用于加速解码
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存的之前kv状态
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 残差
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 拼接输出
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        # (hidden_states, self_attn_weights, present_key_value)
        return outputs


class BuDingPreTrainedModel(PreTrainedModel):
    config_class = BuDingConfig
    # 定义了模型内部子模块命名的基础前缀，当加载或保存模型时，这个前缀将用于识别模型主体部分。
    base_model_prefix = "model"
    # 表明该模型支持梯度检查点技术，这是一种内存优化策略，可减少模型训练时所需的显存
    supports_gradient_checkpointing = True
    # 指定了在序列化过程中不应被拆分的模块列表，即在模型保存与加载时保持这些模块作为一个整体。
    _no_split_modules = ["TinyllmDecoderLayer"]
    # 在跨设备数据移动时，指示哪些关键字（key）对应的数据应该跳过设备放置步骤。
    _skip_keys_device_placement = "past_key_values"
    # Scaled Dot Product Attention (SDPA)
    _supports_sdpa = True
    # 表示模型支持缓存机制，这在自回归模型（如Transformer解码器）中很常见，
    # 用于存储先前计算的结果以加快后续时间步长的计算速度。
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class BuDingModel(BuDingPreTrainedModel):
    """
    定义模型主体
    """
    def __init__(self, config: BuDingConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [BuDingDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = BuDingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,    # 每个输入序列词元在位置嵌入中的位置索引
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 可用于加速序列解码预先计算的隐藏状态（自注意力块和交叉注意力块中的键和值）
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 是否输出decoder layer的结果
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 是否使用kv cache
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 返回采用字典返回
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            # 生成一个从past_key_values_length到seq_length + past_key_values_length的整数序列
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            # 将生成的序列重塑为形状为(1, seq_length)的张量，然后展平为形状为(-1, seq_length)的张量
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            # inputs_embeds shape: [batch_size, seq_len, hidden_size]
            inputs_embeds = self.embed_tokens(input_ids)

        # 适应不同注意力机制对注意力掩码的不同要求而设计的
        if self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for decoder_layer in self.layers:
            # 1.隐藏状态保存
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # 2.梯度检查，方便在反向传播时只激活部分层，节省内存资源
            # 3.解码层：
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            # 4.更新隐藏状态
            hidden_states = layer_outputs[0]
            # 5.更新缓存
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            # 6.注意力输出保存
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # BaseModelOutputWithPast类表示模型输出，基类是OrderedDict
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BuDingForCausalLLM(BuDingPreTrainedModel):
    """
    基于BuDingModel的基础上，添加线性层，将输出的隐藏层维度映射到词表维度
    即logits，shape为: [batch_size, seq_len, vocab_size]
    """
    # 权重共享
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BuDingConfig):
        super().__init__(config)
        self.model = BuDingModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # 对于自回归模型（如GPT系列），我们需要将模型输出的logits向前移动一位，
            # 这样使得模型预测的是当前时刻 t 的下一个词，而非当前词本身
            shift_logits = logits[..., :-1, :].contiguous()
            # 同时，也需要将真实标签（labels）向前移动一位以与调整后的logits对齐
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)

            # 将移位后的 logits 和 labels 扁平化，即将它们展平为一维张量
            # 其中shift_logits变成 (batch_size * sequence_length, vocab_size) 的形式
            # shift_labels变为 (batch_size * sequence_length) 的形式
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            # 确保模型并行计算时，labels的数据存储位置与logits一致
            shift_labels = shift_labels.to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def print_model_parameters(model):
    """
    打印模型各个层参数
    """
    param_sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_sum += param.numel()
            print(f"Layer: {name}, Parameters: {param.numel()}")
    print(f"Total of parameters: {param_sum}")


if __name__ == '__main__':
    args_92m = BuDingConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=1408,
        rope_theta=10000.0,
        max_position_embeddings=1024,
        vocab_size=64798,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BuDingForCausalLLM(args_92m).to(device)

    print_model_parameters(model)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long, device=device)
    labels = torch.tensor([[1, 4, 3], [2, 3, 1]], device=device)

    print(f"input_ids shape: {input_ids.shape}")
    outputs = model(input_ids, labels=labels)
    print(f"logits shape: {outputs.logits.shape}")
    print(outputs.loss)
