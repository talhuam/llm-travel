from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random


class PTDataset(Dataset):
    """
    预训练数据集
    data_processor/pretrain_processor.py进行数据预处理
    """
    def __init__(self, data_path_list, max_length=512):
        """
        :param data_path_list: 数据路径列表
        :param max_length: 样本最大长度
        :param memmap:
        """
        super(PTDataset, self).__init__()

        data_list = []
        for data_path in data_path_list:
            with open(data_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.uint16)
                data_list.append(data)
        # 将所有文件的数据连接成一个大的数组
        data = np.concatenate(data_list)
        # 裁剪数组，使其长度为 max_length 的整数倍
        data = data[:max_length * int(len(data) / max_length)]
        # np.random.shuffle(data)
        self.data = data.reshape(-1, max_length)

        self.shuffle_index = list(range(len(self.data)))
        random.shuffle(self.shuffle_index)
        print("train data.shape:{}".format(self.data.shape))
        print("Data loading is completed.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        index = self.shuffle_index[index]
        sample = self.data[index]
        X = np.array(sample).astype(np.int64)
        # Y = np.array(sample[1:]).astype(np.int64)
        input_ids = torch.LongTensor(X)
        # labels = torch.LongTensor(Y)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


class PTDatasetMap(Dataset):
    """
    数据集类，使用内存映射处理大文件数据，以节省内存。
    """
    def __init__(self, data_path_list: list, max_length=512):
        super(PTDatasetMap, self).__init__()

        self.data = []  # 存储每个文件的内存映射数据
        self.index_map = {}  # 索引映射，便于快速定位样本
        self.token_size = 0  # token数
        self.data_size = 0  # 样本数量

        for idx, file_path in enumerate(data_path_list):
            with open(file_path, 'r') as f:
                nbytes = f.seek(0, 2)
                flen = f.tell() // np.dtype('uint16').itemsize

            # 更新统计信息和索引映射
            self.token_size += flen
            self.index_map.update({self.data_size + i: (idx, i) for i in range(flen // max_length)})
            self.data_size += flen // max_length

            # 使用内存映射读取数据
            self.data.append(np.memmap(file_path, dtype=np.dtype('uint16'), shape=(flen // max_length, max_length)))

        print('total token size: {:,} token,  data sample size: [{:,}, {}]'.format(self.token_size, self.data_size,
                                                                                   max_length))

        # 初始化时生成随机索引列表
        self.shuffled_indices = list(self.index_map.keys())
        random.shuffle(self.shuffled_indices)

        # print(self.shuffled_indices)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index: int):
        real_index = self.shuffled_indices[index]
        fi, i = self.index_map[real_index]
        sample = self.data[fi][i]
        X = np.array(sample).astype(np.int64)
        # Y = np.array(sample[1:]).astype(np.int64)
        input_ids = torch.LongTensor(X)
        # labels = torch.LongTensor(Y)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }