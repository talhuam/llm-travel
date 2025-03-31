import os
import argparse
from huggingface_hub import snapshot_download
# huggingface国内镜像站
# autodl需要注释掉该镜像站，并运行：source /etc/network_turbo
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

def download_model(model_name, local_dir):
    """
    huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./downloaded
    :param model_name:
    :param local_dir:
    :return:
    """
    model_path = os.path.join(local_dir, model_name)
    snapshot_download(repo_id=model_name, local_dir=model_path)
    print(f"模型已下载到 {model_path}")


def download_dataset(dataset_name, local_dir):
    """
    huggingface-cli download --repo-type dataset --resume-download neulab/conala --local-dir ./downloaded
    :param dataset_name:
    :param local_dir:
    :return:
    """
    dataset_path = os.path.join(local_dir, dataset_name)
    snapshot_download(repo_id=dataset_name, local_dir=dataset_path, repo_type="dataset")
    print(f"数据集已下载到 {dataset_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=["model", "dataset"], help='下载类型')
    parser.add_argument('--dir', type=str, default='./download', help='本地文件路径')
    parser.add_argument('--name', type=str, required=True, help='模型或者数据集名称')

    args = parser.parse_args()

    if args.type == "model":
        print(f"downloading model: {args.name}")
        download_model(args.name, args.dir)
    elif args.type == "dataset":
        print(f"downloading dataset: {args.name}")
        download_dataset(args.name, args.dir)