import pandas as pd
import torch
from torch.utils.data import Dataset
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os


class ExcelDataset(Dataset):
    def __init__(self, data_path, transform=None, train=True):
        self.transform = transform
        # self.le = LabelEncoder()
        self.data = []
        self.labels = []
        self.train = train  # 是否加载训练数据集

        # 循环读取所有类别的数据文件
        for label in os.listdir(data_path):
            df = pd.read_excel(os.path.join(data_path, label))
            df = df.dropna(axis=0)  # 删除有缺失值的行
            self.data.append(df.values)
            # 使用文件名作为标签
            self.labels.extend([label] * len(df))

        self.data, self.val_data, self.labels, self.val_labels = train_test_split(self.data, self.labels, test_size=0.2)
        assert len(self.data) == len(self.labels)

    def __len__(self):
        if self.train:
            return len(self.data)
        else:
            return len(self.val_data)

    def __getitem__(self, index):
        if self.train:
            item, label = self.data[index], self.labels[index]
        else:
            item, label = self.val_data[index], self.val_labels[index]

        if self.transform is not None:
            item = self.transform(item)

        return {"inputs": torch.FloatTensor(item), "labels": torch.LongTensor([label])}