import torch
from torch.autograd import Variable
import torch.nn.functional as F
import timm
import numpy as np
from datetime import datetime
import os
from utils import AvgMeter, overall_acc
from dataset import TestDataset
from torch.utils.tensorboard import SummaryWriter
import csv


label_map = {
    'AIS' : 0,
    'MIA' : 1,
    '1'   : 2,  # 贴壁生长型
    '2'   : 3,  # 腺泡和乳头状
    '3'   : 4,  # 微乳头和实性
    }


if __name__ == "__main__":
    """
    需要修改的三个参数:
    weight_path:  模型权重路径
    save_metrics: 保存测试结果路径
    model:        更换模型
    如果需要更改数据集需要在dataset里另行处理
    """
    weight_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_101_SEResNext50_5cls_20240118/epoch_25.pth'
    save_metrics = '/data2/ceiling/workspace/RCFNet/save_model/ROI_101_SEResNext50_5cls_20240118/metrics.csv'
    num_classes = 5
    image_size = (352, 352)
    model = timm.create_model('seresnext50_32x4d', pretrained=True, num_classes=num_classes, 
                              pretrained_cfg_overlay=dict(file='./pretrained_weights/seresnext50_32x4d.bin'))
    json_path = '/data1/ceiling/workspace/Gross/RCFNet/data_split/ROI_combined_101.json'
    root_path = '/data5/ceiling/datasets/Gross-Combined/ROI/'
    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()
    model.eval()
    params = model.parameters()
    print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    test_dataset = TestDataset(root_path = root_path,
                               json_path = json_path,
                               mode = 'test',
                               label_map = label_map,
                               image_size = image_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                              num_workers=4, pin_memory=True)
    print("测试集包含图像数目:", len(test_dataset))

    csv_header = ['step', 'patient name', 'acc', 'label', 'pred']
    csv_rows = []
    total_acc = []
    average_acc = []
    label_list = []
    
    print("*" * 20, "|| 测试开始 ||", "*" * 20)
    print("{}".format(datetime.now()))
    
    # 需要体现类别的准确性
    for step, pack in enumerate(test_loader, start=1):
        image, label, name = pack
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        with torch.no_grad():
            r = model(image)
        cls_acc = overall_acc(r, label)

        label = label.cpu().numpy()[0]
        pred = torch.argmax(r, dim=-1).cpu().numpy()[0]
        cls_acc = cls_acc.cpu().numpy()
        name = name[0]

        print("Step: {:4}/{:4}, 患者姓名: {:6}, 预测准确率: {:1}, ".format(step, len(test_dataset), name[:-4], cls_acc))
        print("Step: {:4}/{:4}, 预测结果: {:2}, 真实结果: {:2}, ".format(step, len(test_dataset), label, pred))
        total_acc.append(cls_acc)
        label_list.append(label)
        csv_rows.append([step, name[:-4], cls_acc, label, pred])
    
    for i in range(num_classes):
        average_acc.append(np.mean(np.array(total_acc)[np.array(label_list) == i]))

    print("测试集整体准确率: {:.4}".format(np.mean(total_acc)))
    print("测试集平均准确率: {:.4}".format(np.mean(average_acc)))

    with open(save_metrics, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header) 
        writer.writerows(csv_rows)
