import torch
import torch.nn as nn
import torch.optim as optim
from envaluedef import evaluate_model
from envaluedef import prob_mc
from sklearn.metrics import confusion_matrix
from pylab import mpl
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import CombinedIMUAndPressureSensorModel
import matplotlib



matplotlib.use('TkAgg')  # 输出使用matplotlib窗口


# 计算模型参数量的函数


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  #if p.requires_grad计算梯度时才包在总数中


# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
# 设置字体 Times New Roman
plt.rc('font', family='Times New Roman')

plt.rcParams.update({
    "font.size": 9.5,                 # 设置全局字体大小
    "font.family": "Times New Roman", # 设置字体为新罗马
    "axes.titlesize": 9.5,            # 图标题字体大小
    "axes.labelsize": 9.5,            # 坐标轴标签字体大小
    "xtick.labelsize": 9.5,           # X 轴刻度字体大小
    "ytick.labelsize": 9.5,           # Y 轴刻度字体大小
    "legend.fontsize": 9.5            # 图例字体大小
})
# 用滑窗截取，形成训练样本集
win_len = 200
step = 50  # 滑窗截取时的步长，小于窗口尺寸使得各个样本间具有联系
class_num = 4  # 分类个数
len_channel = 3  # 通道个数
batchsize = 128  #64

data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
save_fig_path = os.path.join(data_root, 'result_photos')

# 读取数据for dev
data = []
labels = []
#列名 X Gyro	Y Gyro	Z Gyro	X acc	Y acc	Z acc	front	back
all_col_name = ['X Gyro', 'Y Gyro', 'Z Gyro', 'X acc', 'Y acc', 'Z acc', 'front', 'back']  # 总通道
#'data/firstbed/output0.csv'
for device in range(0, class_num):
    file_path = os.path.join(data_root, f'data/thirdbed/output{device}.csv')  # 数据文件路径，在本文件夹上层 dataset里
    #data/secondbed/output{device}.csv
    col_name = all_col_name[0: len_channel]  # 用于训练的通道
    col_idx = list(range(len_channel))
    df = pd.read_csv(file_path, sep=',', header=0, usecols=col_name)
    i = 0
    g_idx = int((len(df) - win_len) / step) + 1
    # 用滑窗截取
    while i < g_idx:
        start = i * step  # 一个窗口起始位置
        end = win_len + i * step  # 结束位置
        data.append(df[start:end])  # 添加一个窗口到样本集中
        i = i + 1
        if class_num == 2:  # 二分类时，只关注正样本和负样本
            if device == 0:
                labels.append(0)
            else:
                labels.append(1)
        else:  # 多分类则为各自样本标签
            labels.append(device)

data = np.array(data)  #有列表变为数组，对数据的处理更加方便
data = data.transpose((0, 2, 1))
print(data.shape)
# 生成的数据大小（g_idx，len_channel，win_len）标签大小（g_idx，）

# 将数据转为Tensor
data = torch.tensor(data, dtype=torch.float32)  # 数据尺寸：
labels = torch.tensor(labels, dtype=torch.long)  # 数据尺寸：
# 按比例划分训练集、验证集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True)

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

# 初始化模型和损失函数
model = CombinedIMUAndPressureSensorModel(class_num, len_channel)  # 模型实例化
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
weight_decay = 0.001  #l2正则化
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)  # Adam优化器

# 训练模型
num_epochs = 50  # 训练回合
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 计算参数量

total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")

# 画图所需参数创建
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
gait_ture = []
gait_pre = []

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_correct = 0
    train_total = 0
    val_correct = 0
    val_total = 0
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, train_predicted = torch.max(outputs, dim=1)
        train_total += batch_y.size(0)
        train_correct += (train_predicted == batch_y).sum().item()
    model.eval()
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_outputs = model(val_x)
            val_loss += criterion(val_outputs, val_y).item()
            _, val_predicted = torch.max(val_outputs, dim=1)
            val_total += val_y.size(0)
            # gait_pre.append(val_predicted).numpy().cpu()
            # gait_ture.append(val_y).numpy().cpu()
            val_correct += (val_predicted == val_y).sum().item()

    # 计算画图参数
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracy = train_correct / train_total
    val_accuracy = val_correct / val_total
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

# 在测试集上评估模型
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for test_x, test_y in test_loader:
        test_x, test_y = test_x.to(device), test_y.to(device)
        test_outputs = model(test_x)
        loss = criterion(test_outputs, test_y)
        test_loss += loss.item()
        _, predicted = torch.max(test_outputs, dim=1)
        gait_pre.append(predicted)
        gait_ture.append(test_y)
        total += test_y.size(0)
        correct += (predicted == test_y).sum().item()
        gait_pre[-1].cpu().numpy()
        gait_ture[-1].cpu().numpy()

test_loss /= len(test_loader.dataset)
accuracy = correct / total
print(f'Test Loss: {test_loss}, Accuracy: {accuracy}')

gait_pre1_cpu = gait_pre[0].cpu()
gait_pre2_cpu = gait_pre[1].cpu()
gait_pre_concatenated = np.concatenate((gait_pre1_cpu.numpy(), gait_pre2_cpu.numpy()))

gait_ture1_cpu = gait_ture[0].cpu()
gait_ture2_cpu = gait_ture[1].cpu()
gait_ture_concatenated = np.concatenate((gait_ture1_cpu.numpy(), gait_ture2_cpu.numpy()))

#计算混淆矩阵
confu_mat, Acc, Sen,Spe ,PPV,F_score,MCC,ave_acc,ave_spe,ave_Sen= evaluate_model(gait_pre_concatenated, gait_ture_concatenated, class_num)
prob_confu_mat = prob_mc(confu_mat)

def plot_Performance_Evaluation(Acc, Sen,Spe ,PPV,F_score,MCC):
    metrics = ['Spe', 'Acc', 'Sen', 'PPV', 'F_score', 'MCC']
    values = [Spe, Acc, Sen, PPV, F_score, MCC]
    num_classes = np.arange(class_num)

    plt.figure(figsize=(12, 6))

    # 绘制每个类别的柱状图
    for i, metric in enumerate(metrics):
        plt.bar(num_classes + i * 0.15, values[i], width=0.15, label=metric)

    plt.xlabel("Class")
    plt.ylabel("Percentage (%)")
    plt.title("Evaluation Metrics by Class")
    plt.xticks(num_classes + 0.3, [f"Class {i}" for i in num_classes])  # 调整x轴以对齐标签
    plt.legend()
    plt.show()


# 下面两个函数需要时调用，分别为绘制混淆矩阵和绘制预测色图

import matplotlib.pyplot as plt
import seaborn as sns

def plot_con_mat(confusion_matrix):  # 绘制混淆矩阵
    # 标签
    labels = ['Health', 'Severity-2', 'Severity-2.5', 'Severity-3']
    plt.figure(figsize=(5, 5))  # 适当缩小图像尺寸
    # 设置字体
    font = {'family': 'Times New Roman', 'size': 9.5}

    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 9.5, "fontname": "Times New Roman"}, fmt=".2f",
                linewidths=0.2, linecolor='gray')
    plt.title('ST-CNN-Transformer', fontdict=font, pad=5)  # 减小标题与图之间的间距
    plt.xticks(fontsize=9.5, fontfamily='Times New Roman', rotation=45)  # 旋转避免重叠
    plt.yticks(fontsize=9.5, fontfamily='Times New Roman', rotation=0)

    plt.tight_layout()  # 自动调整布局，使其更紧凑
    plt.subplots_adjust(top=0.9, bottom=0.267,left=0.208,right=0.985)
    plt.show()

#画混淆矩阵
plot_con_mat(prob_confu_mat)

#画六个性能指标图
plot_Performance_Evaluation(Acc, Sen,Spe ,PPV,F_score,MCC)
# plot_con_mat(confusion_matrix, os.path.join(save_fig_path, 'confusion_matrix.png'))
# plot_predict(gait_pre)

# 设置窗口大小

plt.rcParams['figure.figsize'] = (10.0, 5.0)
# 画出训练集和校验集的损失
plt.figure(figsize=(5,3))
plt.rcParams['lines.linewidth'] = 1.5
plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val')
plt.xlabel('Epoch', fontsize=9.5)
plt.ylabel('Model Loss', fontsize=9.5)
plt.legend(loc=1)
plt.tight_layout()
plt.subplots_adjust(top=0.944, bottom=0.178, left=0.126, right=0.834)

# 画出每轮训练的正确率
plt.figure(figsize=(5,3))
plt.rcParams['lines.linewidth'] = 1.5
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val')
plt.xlabel('Epoch', fontsize=9.5)  #fontsize用来更改字体大小
plt.ylabel('Accuracy', fontsize=9.5)
plt.legend(loc=4)

plt.tight_layout()
plt.subplots_adjust(top=0.944, bottom=0.178, left=0.126, right=0.834)
plt.show()

#plot_con_mat(confusion_matrix)
# 保存模型
torch.save(model, os.path.join(data_root, 'model.pth'))
