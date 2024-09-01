import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    
    参数:
        y_true (array-like): 真实标签
        y_pred (array-like): 预测标签
        classes (list): 类别列表
        normalize (bool, optional): 是否将混淆矩阵归一化. 默认为 False.
        title (str, optional): 图表标题. 默认为 None.
        cmap (matplotlib colormap, optional): 颜色映射. 默认为 plt.cm.Blues.
    """
    
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("归一化的混淆矩阵")
    else:
        print('未归一化的混淆矩阵')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(cm) -0.5, -0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Prediction')


if __name__ == "__main__":
    """
    需要修改的四个参数:
    metrics_path: 测试结果csv文件的保存路径
    title:        混淆矩阵标题名称
    normalize:    是否对混淆矩阵进行归一化
    save_path:    保存路径
    """
    model_name = 'Coarse with Clinic'
    metrics_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_303_Fusion_clinic_RCFNet_5cls_20240206/metrics.csv'
    title = 'Performance of {}'.format(model_name)
    normalize = False
    save_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_303_Fusion_clinic_RCFNet_5cls_20240206/Confusion_Matrix_{}.png'.format(model_name)
    
    metrics = pd.read_csv(metrics_path)
    # label = 1 * (np.array(list(metrics['label'])) >= 2)
    label = np.array(list(metrics['label']))
    pred = np.array(list(metrics['net1']))
    
    # classes = ['AIS', 'MIA', '1', '2', '3']
    # classes = ['0', '1', '2']
    classes = ['Low', 'High']
    plot_confusion_matrix(label, pred, classes, normalize=normalize, title=title)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print("保存完成:", save_path)