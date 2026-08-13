import numpy as np
import pandas as pd

def evaluate_model(y_pre, y_true, class_num):
    # 初始化存储TP、TN、FP、FN的数组
    T = np.zeros([2, class_num])  # 存储多分类TP、TN T[0]为 TP正样本、T[1]为 TN负样本
    F = np.zeros([2, class_num])  # 存储多分类FP、FN F[0]为 FP正样本、F[1]为 FN负样本
    confu_mat = np.zeros([class_num, class_num])  # 混淆矩阵

    # 计算混淆矩阵和TP、FP、TN、FN
    for i in range(len(y_true)):
        true_class = y_true[i]
        pre_class = y_pre[i]
        if true_class == pre_class:  # 预测正确
            T[0, true_class] += 1  # TP+1
            for j in range(class_num):
                if j != true_class:
                    T[1, j] += 1  # 其他类别的TN+1
            confu_mat[true_class, true_class] += 1
        else:  # 预测错误
            F[0, pre_class] += 1  # FP+1
            F[1, true_class] += 1  # FN+1
            confu_mat[true_class, pre_class] += 1

    # 计算六个指标
    TP = T[0]
    TN = T[1]
    FP = F[0]
    FN = F[1]

    Acc = (TP + TN) / (TP + TN + FP + FN) * 100  # 准确率
    Sen = TP / (TP + FN) * 100  # 敏感性（召回率）
    Spe = TN / (TN + FP) * 100  # 特异性
    PPV = TP / (TP + FP) * 100  # 正预测值
    F_score = 2 * (PPV * Sen) / (PPV + Sen)  # F1分数
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))  # MCC

    ave_acc=np.mean(Acc)
    ave_spe=np.mean(Spe)
    ave_Sen=np.mean(Sen)
    np.set_printoptions(precision=2, suppress=True, floatmode='fixed')  # 保留两位小数，禁用科学计数法

    print("Spe:", Spe)
    print("Acc:", Acc)
    print("Sen:", Sen)
    print("PPV:", PPV)
    print("F_score:", F_score)
    print("MCC:", MCC)
    print("ave_acc",np.mean(Acc))
    print("ave_spe", np.mean(Spe))
    print("ave_Sen", np.mean(Sen))

    return confu_mat, Acc, Sen, Spe, PPV, F_score, MCC,ave_acc,ave_spe,ave_Sen


# 概率矩阵计算函数
def prob_mc(confu_mat):
    col_sums = confu_mat.sum(axis=0)
    prob_matrix = (confu_mat / col_sums) * 100
    # prob_matrix = np.round(prob_matrix, 2)
    return prob_matrix

