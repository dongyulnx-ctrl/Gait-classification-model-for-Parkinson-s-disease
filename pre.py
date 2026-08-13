import torch
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt


# 设置字体 Times New Roman
plt.rc('font', family='Times New Roman')

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(data_root, 'model.pth')
model = torch.load(model_path)
model = model.to(device)
model.eval()  # 设置为评估模式


def preprocess_data(file_path, win_len=40, step=40):
    # 定义与训练时相同的预处理步骤
    # 用滑窗截取，形成训练样本集
    win_len = 240  # 窗口大小，一步在50-60帧之间，设置容纳多步窗口为一个样本
    step = 110  # 滑窗截取时的步长，小于窗口尺寸使得各个样本间具有联系
    class_num = 4  # 分类个数
    len_channel = 7  # 通道个数
    all_col_name = ['xspeed', 'yspeed', 'zspeed', 'xacc', 'yacc', 'zacc', 'front', 'back']
    col_name = all_col_name[0: len_channel]
    df = pd.read_csv(file_path, sep=',', header=0, usecols=col_name)
    data = []
    i = 0
    g_idx = int((len(df) - win_len) / step) + 1
    while i < g_idx:  # 请根据实际需求调整这个循环条件
        start = i * step
        end = win_len + i * step
        data.append(df[start:end])
        i += 1
    data = np.array(data)
    data = data.transpose((0, 2, 1))
    data = torch.tensor(data, dtype=torch.float32)
    return data


def classify(file_path, model, device):
    data = preprocess_data(file_path)  # 预处理数据
    data = data.to(device)  # 确保数据在正确的设备上
    with torch.no_grad():  # 不计算梯度，减少计算开销
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
    return predicted


def plot_predict(gait_pre):
    # 定义不同分类的颜色
    colors = {0: '#76D7C4', 1: '#F7DC6F', 2: '#FFC0CB', 3: '#FF7750'}
    # 创建图形和轴，同时设定fig size参数来「拉长」横轴
    fig, ax = plt.subplots(figsize=(50, 6))  # 12英寸宽，6英寸高

    x = np.linspace(1, len(gait_pre), len(gait_pre))
    # 绘制色块
    for index, (time, result) in enumerate(zip(x, gait_pre)):
        # 色块的底部和顶部边界
        bottom = result
        top = result + 1
        # 色块的左右边界，这里没有改变每个色块的实际长度
        left = time - (x[1] - x[0])/2
        right = time + (x[1] - x[0])/2
        # 绘制色块及其边界线
        ax.fill_betweenx([bottom, top], left, right, color=colors[result])

    # 设定x轴和y轴的范围和标签，保持x轴数据范围为1到10
    ax.set_xlim(1, len(gait_pre))
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['Stage0', 'Stage1', 'Stage2', 'Stage3'], fontsize=14)

    # 设置图表标题和坐标轴标签
    ax.set_title('Time series of Parkinsonian abnormalities and normal gait classification')
    ax.set_xlabel('Time/s', fontsize=14)
    ax.set_ylabel('Classification Result', fontsize=14)

    # 展示图形
    plt.show(block=False)

# 假设有一个新的CSV文件需要进行分类
file_path = os.path.join(data_root, 'dataset', 'predict.csv')
predicted_labels = classify(file_path, model, device).cpu().numpy()
plot_predict(predicted_labels)
print(predicted_labels)
