# ST-CNN-Transformer

基于**时空卷积网络（Spatial-Temporal CNN）+ Transformer** 的帕金森病（Parkinson's disease, PD）步态严重程度分级模型。

## 项目简介

本仓库实现了论文 *《Diagnosis and severity rating of Parkinson's disease based on multimodal gait signal analysis with GLRT and ST-CNN-Transformer networks》* 中的核心算法代码，用于训练一个 **ST-CNN-Transformer** 结构（时空卷积网络 + Transformer）的深度学习模型，对帕金森病患者的步态严重程度进行四分类：**Healthy（健康）/ Severity-2 / Severity-2.5 / Severity-3**（对应扩展 H&Y 分级）。

### 整体流程

```
多模态步态数据（三轴加速度 + 三轴角速度 + 前后脚压力，共 8 通道）
        │
        ├─ 均值滤波（去噪）
        │
        ├─ GLRT 零速点(ZVP)检测 → 步态周期分割与融合
        │
        ├─ ST-CNN 双分支提取时空特征
        │      ├─ 时间流：Conv1D（一维卷积，捕捉时序模式）
        │      └─ 空间流：Conv2D（把 8 通道当"伪图像"，捕捉通道间关系）
        │
        ├─ Transformer 编码器（自注意力，建模步态周期间的长程依赖）
        │
        └─ MLP 分类头 → 输出 4 类严重程度
```

> 详细的方法描述、网络参数（Table Ⅰ）、实验设置与结果（Table Ⅱ~Ⅴ、Fig.1~10）请参考论文正文。本仓库只提供与训练和画图相关的代码。

## 文件说明

### 🧠 模型定义

| 文件 | 作用 |
|---|---|
| **`model.py`** | 定义网络结构。包含 `PositionalEncoding`（Transformer 位置编码）和 `CombinedIMUAndPressureSensorModel`（主模型）。主模型为**双分支结构**：`conv1d` 时间流（Conv1d 8→16→32，kernel=4，MaxPool）+ `conv2d` 空间流（把多通道输入当单通道伪图像做二维卷积），两分支拼接后经位置编码送入 `TransformerEncoder`（3 层、4 头），最后经全局平均池化 + 全连接层（MLP）输出 4 类。 |

### 🏋️ 训练脚本

| 文件 | 作用 |
|---|---|
| **`train.py`** | **主训练脚本**。滑窗截取步态数据生成训练样本（`win_len=200`、`step=50`、3 通道），按 80/20 划分训练/测试集，训练 50 个 epoch（Adam + L2 正则化 + 交叉熵损失）。训练结束后：计算混淆矩阵、画 **6 项评估指标柱状图**、画 **loss / accuracy 曲线**，并保存模型 `model.pth`。 |
| **`有数据增强的train.py`** | 带**数据增强**的训练版本。使用全部 8 通道（`len_channel=8`），并引入 `SeverityMixDataset`——以 0.2 概率对不同严重程度样本做 mixup 式线性混合，增强样本多样性、缓解小样本过拟合。其余流程与 `train.py` 一致。 |
| **`times_train.py`** | **多次自动训练**脚本。循环训练 `num=40` 次，每次记录 `ave_acc / ave_spe / ave_sen` 三个平均指标，累积后保存为 `metrics_results.csv`——这份多次运行的结果正是论文 Fig.9 **箱型图**的数据来源。 |

### 📊 画图相关代码

| 文件 | 作用 | 对应论文图 |
|---|---|---|
| **`zupt.py`** | GLRT 零速检测。对 IMU 数据做偏置校正后计算加速度模 `acc_mod` 和角速度模 `gyro_mod`，用 `zero_v()` 检测零速点(ZVP)，并画出 **Acc Mod / Gyro Mod / Zero Speed Indicator** 三轴叠加图。 | Fig.5 |
| **`period zupt.py`** | 周期零速检测 + 步态周期融合。对 4 个设备/严重程度的数据分别做零速检测，计算步态周期的**均值与方差**，再用 `weighted_fusion_by_variance()`（方差倒数加权）做周期融合，输出融合后的周期均值/方差。 | —（GLRT 周期融合的数值计算） |
| **`pre.py`** | 预处理 + 预测可视化。`preprocess_data()` 滑窗截取（240 窗口），`classify()` 加载已训练模型对 CSV 做预测，`plot_predict()` 用 **4 种颜色色块** 沿时间轴画出分类结果。 | —（预测结果色块图） |

### 🔧 数据与评估工具

| 文件 | 作用 |
|---|---|
| **`envaluedef.py`** | **评估指标计算**。`evaluate_model()` 根据预测/真实标签统计 TP/TN/FP/FN，计算混淆矩阵和 6 项指标：**Acc（准确率）、Sen（敏感性/召回率）、Spe（特异性）、PPV（正预测值）、F-score、MCC**，并输出各指标平均值。`prob_mc()` 把混淆矩阵按列归一化为百分比概率矩阵（用于画混淆矩阵热力图）。 |
| **`gait_dataset.py`** | PyTorch `Dataset` 类（`ExcelDataset`），从 Excel 文件读取数据并按 80/20 划分训练/验证集。早期版本的数据加载方式，主训练已改用滑窗截取。 |
| **`hex2float.py`** | 硬件数据解析小工具：把数据采集设备输出的 HEX 字符串按 4 字节一组解析为 32 位浮点数（`struct.unpack`）。 |

## 环境依赖

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

## 使用方法

1. **修改数据路径**：训练脚本中数据路径是硬编码的绝对路径（形如 `F:\论文\lcy\thirdbed\thirdbed\output{device}.csv`），使用前需改成你自己的数据目录。数据为 8 列 CSV：`X Gyro, Y Gyro, Z Gyro, X acc, Y acc, Z acc, front, back`（前三轴角速度 + 三轴加速度 + 前/后脚压力）。

2. **训练**：
```bash
python train.py                 # 基础训练
python 有数据增强的train.py      # 带数据增强训练
```

3. **多次训练采集箱型图数据**：
```bash
python times_train.py            # 跑 40 次，生成 metrics_results.csv
```

4. **零速检测 / 周期融合**：
```bash
python zupt.py                   # 画零速检测图
python "period zupt.py"          # 步态周期融合
```

## 说明

- 数据采集自襄阳市第一人民医院，实验方案与伦理批准号见论文正文。
- 训练结果为论文中的 98.81% 四分类准确率，具体指标见表 Ⅳ、Ⅴ 及图 7~10。
