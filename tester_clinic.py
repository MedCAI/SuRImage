import torch
from torch.autograd import Variable
import torch.nn.functional as F
import timm
import numpy as np
from datetime import datetime
import os
from utils import AvgMeter, overall_acc
from dataset_with_clinic import TestDataset
from torch.utils.tensorboard import SummaryWriter
import csv
from models import FineNet, CoarseNet, Fusion_clinic
from torchcam.methods import CAM, GradCAM


def attention_crop(input_image, attention_map):
    b, c, h, w = input_image.shape
    ret = []
    for i in range(b):
        cam = attention_map[i, 0] # h w
        loc = torch.argmax(cam)
        row, col = loc % (11), loc // (11)
        if row <= 3:
            row = 3
        elif row >= 7:
            row = 7
        if col <= 3:
            col = 3
        elif col >= 7:
            col = 7
        crop_image = input_image[i:i+1, :, 32*(col-3): 32*(col+4), 32*(row-3): 32*(row+4)]
        crop_image = F.interpolate(crop_image, (h, w), mode='bilinear')
        ret.append(crop_image)
    ret = torch.stack(ret, dim=0)
    return ret.squeeze(1)


label_map = {
    'AIS' : 0,
    'MIA' : 1,
    '1'   : 2,
    '2'   : 3,
    '3'   : 4,
    }

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
    weight1_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_404_RCFNet_5cls_20240121/net1_epoch_35.pth'
    weight2_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_404_RCFNet_5cls_20240121/net2_epoch_35.pth'
    weight_model_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_404_Fusion_clinic_RCFNet_5cls_20240206/fusion_epoch_1.pth'
    save_metrics = '/data2/ceiling/workspace/RCFNet/save_model/ROI_404_Fusion_clinic_RCFNet_5cls_20240206/metrics.csv'
    json_path = '/data1/ceiling/workspace/Gross/RCFNet/data_split/ROI_combined_404.json'
    root_path = '/data3/ceiling/datasets/Gross-Combined/ROI/'
    clinic_path = '/data1/ceiling/workspace/Gross/RCFNet/excels/GDPH_image_clinic2024_model.csv'
    
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
    model = Fusion_clinic(in_channels=4096, num_clinic=9, num_features=256)
    model = model.cuda()
    model.load_state_dict(torch.load(weight_model_path))
    print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.eval()
    
    test_dataset = TestDataset(root_path = root_path,
                               json_path = json_path,
                               mode = 'test',
                               label_map = label_map,
                               name_length=4,
                               clinic_path = clinic_path,
                               clinic_info = ['age', 'sex', 'weight', 'height', 'BMI', 'smooking', 'smooking frequency', 'smooking history', 'maximum diameter'],
                               image_size = image_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                              num_workers=4, pin_memory=True)
    print("测试集包含图像数目:", len(test_dataset))

    csv_header = ['step', 'patient name', 'label', 'net1', 'net2', 'acc1', 'acc2']
    csv_rows = []
    total_acc1 = []
    total_acc2 = []
    average_acc1 = []
    average_acc2 = []
    label_list = []
    
    print("*" * 20, "|| 测试开始 ||", "*" * 20)
    print("{}".format(datetime.now()))
    
    # 需要体现类别的准确性
    for step, pack in enumerate(test_loader, start=1):
        image, clinic, label, name = pack
        image = image.cuda()
        clinic = clinic.cuda()
        label_coarse = 1 * (label >= 2)
        label_coarse = label_coarse.long().cuda()
        
        label_fine = label.long().cuda()
        label = label.long().cuda()
        
        with torch.no_grad():
            r1 = net1.classifier[:-1](image)
            r2 = net2.classifier[:-1](image)
            c, f = model(torch.cat([r1, r2], dim=-1), clinic)
        cls_acc1 = overall_acc(c, label_coarse)
        cls_acc2 = overall_acc(f, label_fine)

        pred1 = torch.argmax(c, dim=-1)
        cls_acc1 = cls_acc1.cpu().numpy()
        name = name[0]
        
        pred2 = torch.argmax(f, dim=-1)
        cls_acc2 = cls_acc2.cpu().numpy()

        print("Step: {:4}/{:4}, 患者姓名: {:6}, 预测准确率: {:1}, ".format(step, len(test_dataset), name[:-4], cls_acc1))
        print("Step: {:4}/{:4}, 真实结果: {:2}, 粗糙结果: {:2}, 精细结果: {:2}".format(step, len(test_dataset), label_fine.cpu().numpy()[0], pred1.cpu().numpy()[0], pred2.cpu().numpy()[0]))
        
        total_acc1.append(cls_acc1)
        total_acc2.append(cls_acc2)
        
        label = label.cpu().numpy()[0]
        label_list.append(label)
        
        csv_rows.append([step, name[:-4], label, pred1.cpu().numpy()[0], pred2.cpu().numpy()[0], cls_acc1, cls_acc2])
        # csv_rows.append([step, name[:-4], Fine_list[label], Coarse_list[pred1.cpu().numpy()[0]], Fine_list[pred2.cpu().numpy()[0]], cls_acc1, cls_acc2])

    for i in range(num_classes):
        average_acc1.append(np.mean(np.array(total_acc1)[np.array(label_list) == i]))
    for i in range(num_classes):
        average_acc2.append(np.mean(np.array(total_acc2)[np.array(label_list) == i]))

    print("测试集Coarse整体准确率: {:.4}".format(np.mean(total_acc1)))
    print("测试集Coarse平均准确率: {:.4}".format(np.mean(average_acc1)))
    print(average_acc1)
    print("测试集Fine整体准确率: {:.4}".format(np.mean(total_acc2)))
    print("测试集Fine平均准确率: {:.4}".format(np.mean(average_acc2)))
    print(average_acc2)

    with open(save_metrics, "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
