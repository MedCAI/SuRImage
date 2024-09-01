import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

    
if __name__ == "__main__":
    """
    需要修改的四个参数:
    metrics_path: 测试结果csv文件的保存路径
    title:        混淆矩阵标题名称
    normalize:    是否对混淆矩阵进行归一化
    save_path:    保存路径
    """
    model_name = 'All Metrics'
    metrics_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_101_SEResNext50_5cls_20240118/metrics.csv'
    title = 'Performance of {}'.format(model_name)
    normalize = False
    save_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_101_SEResNext50_5cls_20240118/Confusion_Matrix_{}.png'.format(model_name)
    
    metrics = pd.read_csv(metrics_path)
    # label = 1 * (np.array(list(metrics['label'])) >= 2)
    label = np.array(list(metrics['label']))
    pred = np.array(list(metrics['net2']))
    
    classes = ['AIS', 'MIA', '1', '2', '3']
    # classes = ['0', '1', '2']
    # classes = ['Low', 'High']
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(label, pred)

    # 计算敏感性和特异性
    true_positives = np.diag(conf_matrix)
    false_positives = np.sum(conf_matrix, axis=0) - true_positives
    false_negatives = np.sum(conf_matrix, axis=1) - true_positives
    true_negatives = np.sum(conf_matrix) - (true_positives + false_positives + false_negatives)
    
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    ppv = true_positives / (true_positives + false_positives)
    npv = true_negatives / (true_negatives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    plr = sensitivity / (1 - specificity)
    nlr = (1 - sensitivity) / specificity
    
    print("True Positives:", true_positives)
    print("False Positives:", false_positives)
    print("False Negatives:", false_negatives)
    print("True Negatives:", true_negatives)
    
    print("Sensitivity (TPR):", sensitivity)
    print("Specificity (TNR):", specificity)
    print("Positive Prediction Value (PPV):", ppv)
    print("Negative Prediction Value (NPV):", npv)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("Positive Likelihood Ratio (PLR):", plr)
    print("Negative Likelihood Ratio (NLR):", nlr)