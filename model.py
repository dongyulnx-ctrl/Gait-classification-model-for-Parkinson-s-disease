import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码
    d_model: 特征维度
    max_len: 序列长度
    dropout:是为了防止对位置编码太敏感
    """

    def __init__(self, d_model, max_len=5000, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, d_model))
        position = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        X = position / div_term
        # P[[:, :, 0::2]这个用法,就是从0开始到最后面,步长为2,代表的是偶数位置
        self.P[:, :, 0::2] = torch.sin(X)
        # self.P[:, :, 1::2] = torch.cos(X)
        if d_model % 2 == 0:
            self.P[:, :, 1::2] = torch.cos(X)
        else:
            self.P[:, :, 1::2] = torch.cos(X)[:, :-1]

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
##
# d_mod = 32,
# input_dim = 32,
# num_heads = 2,
# hidden_dim = 128,
# num_layers = 3,
# dropout = 0.25):  # 编码层的遗忘率

class CombinedIMUAndPressureSensorModel(nn.Module):
    def __init__(self, out_dim, len_channel,
                 d_mod=32,#32
                 input_dim=32,#32
                 num_heads=2,#2
                 hidden_dim=256,#原来参数256
                 num_layers=3,#3
                 dropout=0.25):#编码层的遗忘率0.25
        super(CombinedIMUAndPressureSensorModel, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=len_channel, out_channels=16, kernel_size=4),  #计算完（64，16，278）
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  #默认步长和窗口长度一致 #计算完（64，16，139）
            nn.Conv1d(in_channels=16, out_channels=input_dim, kernel_size=4),  #计算完（64，32，136）
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  #（64，32，68） 默认步长和窗口长度一致
        )
        self.conv2d = nn.Sequential(
            # nn.Conv2d(in_channels=1,out_channels=input_dim,kernel_size=(5,5)),#一层conv2d
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 2)),  # (64 16 6,278)  两层conv2d
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1),  # (64 16 3 139)
            # (64 32 2 139)
            nn.Conv2d(in_channels=16, out_channels=input_dim, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # (64 32 1 69)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.positional_encoding = PositionalEncoding(d_mod, dropout=0)
        encoder_layer = nn.TransformerEncoderLayer(d_mod, num_heads, hidden_dim, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(input_dim,128 )  #输入的特征为137*32   128
        self.fc2 = nn.Linear(128, out_dim) #2192  升维6321
        self.dropout = nn.Dropout(0.2)  #连接层的遗忘率   #0.2

    def forward(self, x):
        x_2d = x.unsqueeze(1)  #（64，8，280）变为（64，1，8，280）   unsqueeze函数，在序号（）之前添加一个维度
        x_1d = self.conv1d(x)
        x_2d = self.conv2d(x_2d)  #变为（64，32，1，69）
        x_2d = x_2d.squeeze(2)  #变为(63，32，69)    squeeze移除维度（）
        combined_output = torch.cat((x_1d, x_2d), dim=2)  #拼接为（64，32  ，137）（bitch,chananle,length）
        combined_output = self.dropout(combined_output)
        combined_output = combined_output.permute(0, 2, 1)
        combined_output = self.positional_encoding(combined_output)
        combined_output = combined_output.permute(1, 0, 2)
        combined_output = self.encoder(combined_output)
        combined_output = combined_output.permute(1, 2, 0)
        combined_output = self.global_avg_pool(combined_output)
        combined_output = combined_output.squeeze(-1)
        combined_output = self.fc1(combined_output)
        combined_output = self.dropout(combined_output)
        combined_output = torch.relu(combined_output)
        combined_output = self.fc2(combined_output)
        return combined_output


        # combined_output = self.encoder(combined_output)  原来的编码器后续的输出部分
        # combined_output = combined_output.permute(1, 0, 2)
        # combined_output = combined_output.contiguous().view(x.size(0), -1)
        # combined_output = self.fc1(combined_output)
        # combined_output = self.dropout(combined_output)
        # combined_output = torch.relu(combined_output)
        # combined_output = self.fc2(combined_output)