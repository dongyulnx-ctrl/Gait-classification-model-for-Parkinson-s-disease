import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os


# 定义零速检测函数
def zero_v(u):
    g = 9.8  # 根据当地重力加速度进行修正
    sigma_acc = 0.03  # 调整acc和gyro的方差
    sigma_gryo = 0.2 * math.pi / 180
    sigma_acc2 = sigma_acc ** 2
    sigma_gryo2 = sigma_gryo ** 2
    W = 10  # 窗口长度
    N = u.shape[0]  # 获得数组U的长度
    T = np.zeros(N - W + 1)  # 创建一个长度为N-W+1的向量，初始元素均为0
    for k in range(N - W + 1):
        ya_m = np.mean(u[k:k + W, :3], axis=0)  # 计算窗口内加速度的均值
        for l in range(k, k + W):
            tmp = u[l, :3] - g * ya_m / np.linalg.norm(ya_m)  # 计算给个点的加速度与窗口加速度均值之间的差异
            T[k] += np.dot(u[l, 3:6], u[l, 3:6]) / sigma_gryo2 + np.dot(tmp, tmp) / sigma_acc2
    T /= W
    gamma = 1e8  # 原来为1e6
    zupt = np.zeros(u.shape[0])
    for k in range(T.shape[0]):
        if T[k] < gamma:
            end_index = min(k + W, len(zupt))  # 确保不超出索引范围
            zupt[k:end_index] = 1
    T = np.concatenate(([np.max(T)], T, [np.max(T)]))
    return zupt

def find_consecutive_ones(arr):  # 找到连续的1
    start_indices = np.where(np.diff(arr) == 1)[0] + 1
    end_indices = np.where(np.diff(arr) == -1)[0]
    if arr[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    if arr[-1] == 1:
        end_indices = np.append(end_indices, len(arr) - 1)
    return list(zip(start_indices, end_indices))

def calculate_start_start_intervals(zupt):  # 获得步态周期
    intervals = []
    sequences = find_consecutive_ones(zupt)
    for i in range(1, len(sequences)):
        start1, _ = sequences[i - 1]
        start2, _ = sequences[i]
        intervals.append(start2 - start1)
    return intervals

def calculate_and_filter_intervals(zupt):  # 滤波，剔除掉误判点
    intervals = calculate_start_start_intervals(zupt)
    filtered_intervals = [interval for interval in intervals if interval >= 30 and interval<=200]  # 过滤掉间隔少于30的数据点
    average_interval = np.mean(filtered_intervals) if filtered_intervals else 0
    return filtered_intervals, average_interval

#创建均值变量和方差变量
mean = np.ones((1, 4))
var = np.ones((1,4))


data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
save_fig_path = os.path.join(data_root, 'result_photos')
# 读取数据for dev
data = []
labels = []
#列名 X Gyro	Y Gyro	Z Gyro	X acc	Y acc	Z acc	front	back
all_col_name = ['X Gyro', 'Y Gyro', 'Z Gyro', 'X acc', 'Y acc', 'Z acc', 'front', 'back']  # 总通道

num=4
for device in range(0, num):
    file_path = os.path.join(data_root,f'C:/Users\86159\Desktop\论文\lcy/thirdbed/thirdbed/output{device}.csv')  # 数据文件路径，在本文件夹上层 dataset里
    col_name = all_col_name[0: 6]  # 用于训练的通道
    col_idx = list(range(6))
    df = pd.read_csv(file_path, sep=',', header=0, usecols=col_name)
    X_Gyro = df['X Gyro'].values + 0.18353
    Y_Gyro = df['Y Gyro'].values - 2.63720
    Z_Gyro = df['Z Gyro'].values + 0.08567
    X_Accl = (df['X acc'].values - 0.00334) * 9.8
    Y_Accl = (df['Y acc'].values + 0.10527) * 9.8
    Z_Accl = (df['Z acc'].values - 0.14068) * 9.8

    acc_mod = np.sqrt(X_Accl ** 2 + Y_Accl ** 2 + Z_Accl ** 2)
    gyro_mod = np.sqrt(X_Gyro ** 2 + Y_Gyro ** 2 + Z_Gyro ** 2)

    u = pd.DataFrame({
        'ax': X_Accl,
        'ay': Y_Accl,
        'az': Z_Accl,
        'gx': X_Gyro,
        'gy': Y_Gyro,
        'gz': Z_Gyro
    }).values
    zupt = zero_v(u)
    filtered_intervals, average_interval = calculate_and_filter_intervals(zupt)
    mean[0,device] = np.mean(filtered_intervals)
    var [0,device]= np.var(filtered_intervals)
    print(f'第{device} 均值')
    print(mean[0,device])
    print(f'第{device} 方差')
    print(var[0,device])


# 定义基于方差的置信度融合函数
def weighted_fusion_by_variance(mean, var):
    # 计算权重为方差的倒数
    weights = 1 / var
    # 归一化权重
    weights /= np.sum(weights)
    # 进行加权平均融合
    fused_mean = np.sum(mean * weights)
    fused_var = np.sum(var * weights ** 2)

    return fused_mean, fused_var


# 对每个设备的方差进行融合
fused_mean, fused_var = weighted_fusion_by_variance(mean, var)

print("融合后的均值：", fused_mean)
print("融合后的方差：", fused_var)
