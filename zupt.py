import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# 读取数据
file_path = ('C:/Users/86159/Desktop/论文/lcy/firstbed/firstbed/output3.csv')
col =['X Gyro','Y Gyro','Z Gyro', 'X acc','Y acc','Z acc']
df = pd.read_csv(file_path, header=0,usecols=col)

# 数据处理
X_Gyro = df['X Gyro'].values + 0.18353
Y_Gyro = df['Y Gyro'].values - 2.63720
Z_Gyro = df['Z Gyro'].values + 0.08567
X_Accl = (df['X acc'].values - 0.00334)*9.8
Y_Accl = (df['Y acc'].values + 0.10527)*9.8
Z_Accl = (df['Z acc'].values - 0.14068)*9.8

# 计算模
acc_mod = np.sqrt(X_Accl**2 + Y_Accl**2 + Z_Accl**2)
gyro_mod = np.sqrt(X_Gyro**2 + Y_Gyro**2 + Z_Gyro**2)

# 零速检测代码（这里假设zero_v函数已经定义好）
u = pd.DataFrame({
    'ax':X_Accl,
    'ay':Y_Accl,
    'az':Z_Accl,
    'gx':X_Gyro,
    'gy':Y_Gyro,
    'gz':Z_Gyro
}).values
def zero_v(u):
    g =9.8 #根据当地重力加速度进行修正
    sigma_acc =0.03   #进行调参，调整这个acc和gyro的方差
    sigma_gryo =0.2*math.pi/180
    sigma_acc2=sigma_acc**2
    sigma_gryo2=sigma_gryo**2
    W=10      #窗口长度
    N =u.shape[0]   #获得数组U的长度   shape[0]获得行的长度
    T = np.zeros(N-W+1)     #创建一个长度为N-W+1  的向量，初始元素均为0
    for k in range(N - W + 1):    #每次迭代只处理一个窗口内的数据    k从0到N-W
        ya_m = np.mean(u[k:k + W, :3], axis=0) # 计算窗口内加速度的均值
        for l in range(k, k + W):  #处理每个窗口内的数据点
            # 计算 tmp
            tmp = u[l, :3] - g * ya_m / np.linalg.norm(ya_m) #计算给个点的加速度与窗口加速度均值之间的差异
                                                            #np.linalg.norm（）用来取模
            # 更新 T 数组
            T[k] += np.dot(u[l, 3:6], u[l, 3:6]) / sigma_gryo2 + np.dot(tmp, tmp) / sigma_acc2 #

    # 将 T 除以窗口大小 W
    T /= W
    gamma = 1e8 #原来为1e6
    zupt = np.zeros(u.shape[0])
    for k in range(T.shape[0]):

        if T[k] < gamma:
            end_index = min(k + W, len(zupt))  # 确保不超出索引范围
            zupt[k:end_index] = 1
    # 更新 T 数组，固定统计数据的边缘
    T = np.concatenate(([np.max(T)], T, [np.max(T)]))
    return zupt


def find_consecutive_ones(arr):    #找到连续的1
    # 使用numpy的diff函数找出数组中从0变为1的位置
    start_indices = np.where(np.diff(arr) == 1)[0] + 1
    # 找出数组中从1变为0的位置
    end_indices = np.where(np.diff(arr) == -1)[0]

    # 如果数组的第一个元素是1，将其索引添加到开始索引中
    if arr[0] == 1:
        start_indices = np.insert(start_indices, 0, 0)
    # 如果数组的最后一个元素是1，将其索引添加到结束索引中
    if arr[-1] == 1:
        end_indices = np.append(end_indices, len(arr) - 1)

    # 将开始和结束索引组合成元组，并返回这些元组的列表
    return list(zip(start_indices, end_indices))

def calculate_start_start_intervals(zupt): #获得步态周期
    intervals = []
    sequences = find_consecutive_ones(zupt)
    for i in range(1, len(sequences)):
        start1, _ = sequences[i-1]
        start2, _ = sequences[i]
        intervals.append(start2 - start1)  # 计算两个开始之间的间隔
    return intervals

def calculate_and_filter_intervals(zupt):  #滤波，剔除掉误判点
    intervals = calculate_start_start_intervals(zupt)
    # 过滤掉间隔少于30的数据点
    filtered_intervals = [interval for interval in intervals if interval >= 30 and interval<=200]
    # 计算平均值
    average_interval = np.mean(filtered_intervals) if filtered_intervals else 0
    return filtered_intervals, average_interval



zupt = zero_v(u)
filtered_intervals, average_interval = calculate_and_filter_intervals(zupt)
print(len(zupt))
print(filtered_intervals)
print(average_interval)
# start_start_intervals = calculate_start_start_intervals(zupt)
# print(start_start_intervals)
# 创建图表
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制加速度模
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Acc Mod', color=color)
ax1.plot(acc_mod, label='Acc Mod', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 创建第二个y轴
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Gyro Mod', color=color)
ax2.plot(gyro_mod, label='Gyro Mod', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# 绘制零速检测的步骤
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.2))  # 将第三个y轴移动到右边
color = 'tab:green'
ax3.set_ylabel('Zero Speed Indicator', color=color)
ax3.plot(zupt, label='Zero Speed Indicator', color=color, drawstyle='steps-post')
ax3.tick_params(axis='y', labelcolor=color)

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax3.legend(loc='lower right')

# 显示图表
plt.title('Acc Mod, Gyro Mod, and Zero Speed Detection')
plt.show()