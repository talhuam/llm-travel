import json
import os
import glob
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd


def process_wiki_clean(file_path):
    """
    https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered
    """
    print("Processing wiki clean...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_tokens = []
    for line in tqdm(data):
        text = line['completion']
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.special_tokens['<eos>'])
        if len(tokens) > 5:
            all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    output_file_path = os.path.join(target_path, base_name + '.bin')
    with open(output_file_path, 'wb') as f:
        f.write(arr.tobytes())


def process_zhihu(input_dir):
    """
    https://huggingface.co/datasets/wangrui6/Zhihu-KOL
    """
    print("Processing zhihu kol...")
    # df = pd.read_parquet("zhizhu/train-00000-of-00005-a1278ede4e8c5cdb.parquet")
    # responses = df['RESPONSE']
    # print(len(responses))
    # print(responses[4000])
    all_tokens = []
    # 使用glob找出文件夹下所有的.parquet
    for file in glob.glob(os.path.join(input_dir, '*.parquet')):
        print(file)
        # 读取parquet文件
        df = pd.read_parquet(file)

        # 提取RESPONSE列
        responses = df['RESPONSE']

        for text in tqdm(responses):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens['<eos>'])
            if len(tokens) > 5:
                all_tokens += tokens
    arr = np.array(all_tokens, dtype=np.uint16)
    output_file_path = os.path.join(target_path, "zhihu" + '.bin')
    with open(output_file_path, 'wb') as f:
        f.write(arr.tobytes())


def process_tigerbot_part(input_dir):
    """
    https://huggingface.co/datasets/TigerResearch/pretrain_zh
    """
    print("Processing tigerBot pretrain zh...")
    all_tokens = []
    file_idx = 0
    # 使用glob找出文件夹下所有的.parquet
    for file in glob.glob(os.path.join(input_dir, '*.parquet')):
        print(file)
        # 读取parquet文件
        df = pd.read_parquet(file)

        # 提取RESPONSE列
        responses = df['content']

        for text in tqdm(responses):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.special_tokens['<eos>'])
            if len(tokens) > 5:
                all_tokens += tokens

            if len(all_tokens) > 400000000:
                arr = np.array(all_tokens, dtype=np.uint16)
                output_file_path = os.path.join(target_path, "tigerbot_part_" + str(file_idx) + '.bin')
                with open(output_file_path, 'wb') as f:
                    f.write(arr.tobytes())

                all_tokens = []
                file_idx += 1

    if len(all_tokens) > 0:
        arr = np.array(all_tokens, dtype=np.uint16)
        output_file_path = os.path.join(target_path, "tigerbot_part_" + str(file_idx) + '.bin')
        with open(output_file_path, 'wb') as f:
            f.write(arr.tobytes())


def process_tigerbot_wiki(input_dir):
    """
    https://huggingface.co/datasets/TigerResearch/tigerbot-wiki-plugin
    """
    print("Processing tigerBot wiki...")
    for subdir, dirs, files in os.walk(input_dir):
        for idx, file in enumerate(files):
            # 只处理txt文件
            if file.endswith('.json'):
                # 获取当前文件的绝对路径
                file_path = os.path.join(subdir, file)
                all_tokens = []
                # 读取jsonl文件
                with open(file_path, 'r', encoding='utf-8') as infile:
                    filee_content = infile.read()

                content_json = json.loads(filee_content)
                for ele in content_json:
                    text = ele['content'].replace("\n", "")
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    tokens.append(tokenizer.special_tokens['<eos>'])
                    if len(tokens) > 5:
                        all_tokens += tokens

                arr = np.array(all_tokens, dtype=np.uint16)
                # 获取最后一级文件名(包含扩展名)
                file_path = os.path.basename(file_path)
                base_name, ext = os.path.splitext(file_path)

                output_file_path = os.path.join(target_path, base_name + '.bin')
                with open(output_file_path, 'wb') as f:
                    f.write(arr.tobytes())


def process_baidu_baike(input_path):
    """
    https://huggingface.co/datasets/xuqinyang/BaiduBaike-5.63M
    """
    print("Processing baidu baike...")
    batch_size = 1000000

    cnt = 0
    batch_cnt = 0
    doc_ids = []

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        line = json.loads(line)
        text = ''
        try:
            text += line['title'] + '：' + line['summary']
        except:
            pass
        for per in line['sections']:
            text += per['title'] + '：' + per['content'] + '。'
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id) > 5:
            doc_ids += text_id
        cnt += 1
        if cnt % batch_size == 0:
            batch_cnt += 1
            arr = np.array(doc_ids, dtype=np.uint16)
            doc_ids = []
            print('cnt:', cnt, 'arr_shape:', arr.shape)
            with open(os.path.join(target_path, './baidubaike_563w_{}.bin'.format(batch_cnt)), 'wb') as f2:
                f2.write(arr.tobytes())
            del arr

    if not doc_ids:
        batch_cnt += 1
        arr = np.array(doc_ids, dtype=np.uint16)
        print('cnt:', cnt, 'arr_shape:', arr.shape)
        with open(os.path.join(target_path, './baidubaike_563w_{}.bin'.format(batch_cnt)), 'wb') as f:
            f.write(arr.tobytes())


def merge_bin(data_path_list: list):
    """
    合并所有bin文件
    """
    data_arr = []
    for data_path in tqdm(data_path_list):
        with open(data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_arr.append(data)
    arr = np.concatenate(data_arr)
    print(arr.shape)
    with open('./data/pretrain_data.bin', 'wb') as f:
        f.write(arr.tobytes())


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("../tokenizer", trust_remote_code=True)
    print(tokenizer.special_tokens)

    corpus_root_path = "E:\\llm\\datasets\\buding_corpus"
    target_path = "E:\\llm\\datasets\\buding_pretrain"
    process_wiki_clean(
        os.path.join(corpus_root_path, "pleisto/wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json"),
    )
    process_zhihu(os.path.join(corpus_root_path, "wangrui6/Zhihu-KOL/data"))
    process_tigerbot_part(os.path.join(corpus_root_path, "TigerResearch/pretrain_zh/data"))
    process_tigerbot_wiki(os.path.join(corpus_root_path, "TigerResearch/tigerbot-wiki-plugin"))
    process_baidu_baike(os.path.join(corpus_root_path, "xuqinyang/BaiduBaike-5.63M/563w_baidubaike.json"))
