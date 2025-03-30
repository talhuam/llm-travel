from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class BuDingConfig(PretrainedConfig):
    """BuDing语言模型 配置文件"""
    model_type = "buding"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            vocab_size=64797,               # 词表大小
            hidden_size=4096,               # 隐藏层大小
            intermediate_size=11008,        # mlp先升维再降维，qwen2该值默认为22016
            num_hidden_layers=32,           # decoder layer堆叠层数
            num_attention_heads=32,         # query注意力头数
            num_key_value_heads=32,         # key/value头数
            hidden_act="silu",              # 激活函数
            max_position_embeddings=2048,
            initializer_range=0.02,         # 用于模型参数初始化，标准差
            rms_norm_eps=1e-6,              # 用于layer norm
            use_cache=True,
            pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
            tie_word_embeddings=False,
            rope_theta=10000.0,             # \theta基数，\theta=10000 ** (-2i/d)
            attention_dropout=0.0,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
