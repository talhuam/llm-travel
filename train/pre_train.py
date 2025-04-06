import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import argparse
import datasets
import torch
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from buding_dataset import PTDataset
from configuration_buding import BuDingConfig
from modeling_budingllm import BuDingForCausalLLM


@dataclass
class ModelArguments:
    """
    模型相关参数
    """
    hidden_size: Optional[int] = field(
        default=512,
        metadata={"help": "hidden_size"}
    )

    num_hidden_layers: Optional[int] = field(
        default=8,
        metadata={"help": "num_hidden_layers"}
    )

    num_attention_heads: Optional[int] = field(
        default=8,
        metadata={"help": "transformer num_attention_heads"}
    )

    intermediate_size: Optional[int] = field(
        default=1408,
        metadata={"help": "intermediate_size"}
    )

    rope_theta: Optional[float] = field(
        default=10000.0,
        metadata={"help": "rope_theta"}
    )

    max_position_embeddings: Optional[int] = field(
        default=1024,
        metadata={"help": "max_position_embeddings"}
    )

    vocab_size: Optional[int] = field(
        default=64798,
        metadata={"help": "vocab_size, ref https://github.com/THUDM/ChatGLM3/issues/634"}
    )


@dataclass
class ScriptArguments:
    """
    其他相关参数
    """
    mode: Optional[str] = field(
        default="pt",
        metadata={"help": "train mode"}
    )

    dataset_dir_or_path: Optional[str] = field(
        default="data/pre_train",
        metadata={"help": "save pretrain *bin file dir"}
    )

    resume: Optional[bool] = field(
        default=False,
        metadata={"help": "use PyTorch 2.0 to compile the model to be faster"}
    )

    base_model_path: Optional[str] = field(
        default=" ",
        metadata={"help": "SFT train, the base model path"}
    )


def data_collator_fn(examples):
    # 将所有样本的输入 (`X`) 和标签 (`Y`) 分别堆叠
    input_ids = torch.stack([example[0] for example in examples])
    labels = torch.stack([example[1] for example in examples])

    # 返回一个字典，包含模型需要的键和值
    data_dict = {
        "input_ids": input_ids,
        "labels": labels
    }
    return data_dict


logger = logging.getLogger(__name__)


def main():
    # 解析命令行参数，在此封装成json的形式，从json配置从解析到dataclass
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, required=True)
    args = parser.parse_args()
    train_args_file = args.train_args_file

    parser = HfArgumentParser((ModelArguments, ScriptArguments, TrainingArguments))
    # 解析--args的形式的命令行参数
    # model_args, script_args, training_args = parser.parse_args_into_dataclasses()
    model_args, script_args, training_args = parser.parse_json_file(json_file=train_args_file)

    # logger format
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.WARN,  # if training_args.local_rank in [-1, 0] else logging.WARN,
                        handlers=[logging.StreamHandler(sys.stdout)], )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    set_seed(training_args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # init_model, attention为MHA
    buding_config = BuDingConfig(**dict(
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        num_key_value_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        rope_theta=model_args.rope_theta,
        max_position_embeddings=model_args.max_position_embeddings,
        vocab_size=model_args.vocab_size,  # 64798
    ))
    model = BuDingForCausalLLM(buding_config)
    model.to(device)

    ########statistic########
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params}, {total_params / 2 ** 20:.2f}M params")
    logger.info(f"可训练参数: {trainable_params}")
    ########statistic########

    def get_bin_files_abs_paths(directory):
        bin_files_paths = []
        for root, dirs, files in os.walk(directory):  # 遍历目录下所有的目录和文件
            for file in files:
                if file.endswith('.bin'):
                    bin_files_paths.append(os.path.abspath(os.path.join(root, file)))
        return bin_files_paths

    data_path_list = get_bin_files_abs_paths(script_args.dataset_dir_or_path)
    if len(data_path_list) == 0:
        logger.error("***************NO INPUT DATA********************")

    train_ds = PTDataset(data_path_list, max_length=model_args.max_position_embeddings)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        # eval_dataset = None,
        # data_collator = data_collator_fn,
    )

    # Training
    trainer.train(script_args.resume)
    torch.save(model.state_dict(), '{}/last_model.pth'.format(training_args.output_dir))
    last_model_dir = os.path.join(training_args.output_dir, 'last_pt_model')
    os.makedirs(last_model_dir, exist_ok=True)
    # https://github.com/huggingface/transformers/issues/28630
    model.save_pretrained(last_model_dir, safe_serialization=False)


if __name__ == "__main__":
    main()
