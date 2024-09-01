from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import timm
from datetime import datetime
import os
from utils import AvgMeter, overall_acc
from dataset import TestDataset
from torch.utils.tensorboard import SummaryWriter
import csv
from models import CoarseNet, FineNet, Fusion


label_map = {
    'AIS' : 0,
    'MIA' : 0,
    '1'   : 1,  # 贴壁生长型
    '2'   : 1,  # 腺泡和乳头状
    '3'   : 1,  # 微乳头和实性
    }
label_list = ['Low Risk', 'High Risk']

if __name__ == "__main__":
    """
    需要修改的参数:
    weight_path:  模型权重路径
    save_metrics: 保存测试结果路径
    model:        更换模型
    save_path:    保存ROC曲线位置
    title:        ROC曲线命名
    如果需要更改数据集需要在dataset里另行处理
    """
    
    mode = 'test'
    weight_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_ambiguous_RCF/net1_BestAcc.pth'
    save_prob = '/data2/ceiling/workspace/RCFNet/save_model/ROI_ambiguous_RCF/prob_coarse_{}.csv'.format(mode)
    model = CoarseNet()
    save_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_ambiguous_RCF/ROC_coarse_{}.png'.format(mode)
    title = 'ROC of CoarseNet'
    json_path = '/data1/ceiling/workspace/Gross/RCFNet/data_split/ROI_combined_ambiguous.json'
    root_path = '/data3/ceiling/datasets/Gross-Combined/ROI'
    
    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()
    model.eval()
    params = model.parameters()
    method = 'weighted'
    
    print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    
    image_size = (352, 352)

    test_dataset = TestDataset(root_path = root_path,
                               json_path = json_path,
                               mode = 'test',
                               label_map = label_map,
                               image_size = image_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                              num_workers=4, pin_memory=True)
    print("测试集包含图像数目:", len(test_dataset))

    y_true = []
    y_scores = []
    csv_header = ['step', 'patient name', 'label', 'Low Risk', 'High Risk']
    csv_rows = []
    for step, pack in enumerate(test_loader, start=1):
        image, label, name = pack
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        with torch.no_grad():
            r = model(image)
        r = F.softmax(r, dim=-1)
        label = label.cpu().numpy()[0]
        pred = r.cpu().numpy()[0]
        name = name[0]
        y_true.append(label)
        y_scores.append(pred)
        csv_rows.append([step, name[:-4], label, pred[0], pred[1]])
    with open(save_prob, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header) 
        writer.writerows(csv_rows)
    print("Prob保存完成:", save_prob)
    # 将真实标签二值化
    n_classes = len(np.unique(y_true))
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    y_true_bin = np.eye(2)[y_true_bin[:,0]]
    y_scores = np.array(y_scores)
    # 计算每个类别的ROC曲线和AUC值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均和宏平均的ROC曲线和AUC值
    fpr[method], tpr[method], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc[method] = auc(fpr[method], tpr[method])

    # 绘制所有类别的ROC曲线
    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'purple'])  # 颜色循环
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(label_list[i], roc_auc[i]))

    plt.plot(fpr[method], tpr[method],
             label=method+'-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc[method]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print("ROC曲线保存完成:", save_path)