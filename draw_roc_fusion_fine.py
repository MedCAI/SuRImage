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
from models import FineNet, CoarseNet, Fusion


label_map = {
    'AIS' : 0,
    'MIA' : 1,
    '1'   : 2,
    '2'   : 3,
    '3'   : 4,
    }
label_list = ['AIS', 'MIA', '1', '2', '3']

Coarse_list = ['Low Risk', 'High Risk']
Fine_list = ['AIS', 'MIA', '1', '2', '3']

if __name__ == "__main__":
    """
    需要修改的三个参数:
    weight_path:  模型权重路径
    save_metrics: 保存测试结果路径
    model:        更换模型
    如果需要更改数据集需要在dataset里另行处理
    """
    weight1_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_505_RCFNet_5cls_20240121/net1_BestAcc.pth'
    weight2_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_505_RCFNet_5cls_20240121/net2_BestAcc.pth'
    weight_model_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_505_Fusion_RCFNet_5cls_20240121/fusion_epoch_2.pth'
    save_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_505_Fusion_RCFNet_5cls_20240121/ROC_fusion_{}.png'.format('fine')
    title = 'ROC of FusionNet'
    save_metrics = '/data2/ceiling/workspace/RCFNet/save_model/ROI_505_Fusion_RCFNet_5cls_20240121/metrics.csv'
    json_path = '/data1/ceiling/workspace/Gross/RCFNet/data_split/ROI_combined_505.json'
    root_path = '/data5/ceiling/datasets/Gross-Combined/ROI'
    save_prob = '/data2/ceiling/workspace/RCFNet/save_model/ROI_505_Fusion_RCFNet_5cls_20240121/prob_fusion_{}.csv'.format('fine')
    
    method = 'weighted'
    num_classes = 5
    image_size = (352, 352)
    net1 = CoarseNet()
    net1 = net1.cuda()
    net1.load_state_dict(torch.load(weight1_path))
    print("net1 has {} paramerters in total".format(sum(x.numel() for x in net1.parameters())))
    net1.eval()
    net2 = FineNet()
    net2 = net2.cuda()
    net2.load_state_dict(torch.load(weight2_path))
    print("net2 has {} paramerters in total".format(sum(x.numel() for x in net2.parameters())))
    net2.eval()
    model = Fusion()
    model = model.cuda()
    model.load_state_dict(torch.load(weight_model_path))
    print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.eval()
    
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
    csv_header = ['step', 'patient name', 'label', 'pred', 'AIS', 'MIA', '1', '2', '3']
    csv_rows = []
    
    # 需要体现类别的准确性
    for step, pack in enumerate(test_loader, start=1):
        image, label, name = pack
        image = image.cuda()
        label_coarse = 1 * (label >= 2)
        label_coarse = label_coarse.long().cuda()
        
        label_fine = label.long().cuda()
        label = label.long().cuda()
        
        with torch.no_grad():
            r1 = net1.classifier[:-1](image)
            r2 = net2.classifier[:-1](image)
            c, f = model(torch.cat([r1, r2], dim=-1))
        c = torch.softmax(c, dim=1)
        f = torch.softmax(f, dim=1)
        label = label.cpu().numpy()[0]
        pred = f.cpu().numpy()[0]
        name = name[0]
        y_true.append(label)
        y_scores.append(pred)
        csv_rows.append([step, name[:-4], label, torch.argmax(f, dim=-1).cpu().numpy()[0], pred[0], pred[1], pred[2], pred[3], pred[4], ])
    with open(save_prob, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header) 
        writer.writerows(csv_rows)
    print("Prob保存完成:", save_prob)
    # 将真实标签二值化
    n_classes = len(np.unique(y_true))
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
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
             label=method + '-average ROC curve (area = {0:0.2f})'
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