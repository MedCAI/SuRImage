import torch
from torch.autograd import Variable
import torch.nn.functional as F
import timm
import numpy as np
from datetime import datetime
import os
from utils import AvgMeter, overall_acc
from dataset import GrossDataset, ValDataset
from torch.utils.tensorboard import SummaryWriter
from models import CoarseNet, FineNet
from torchcam.methods import CAM, GradCAM
import random


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


def attention_drop(input_image, attention_map):
    b, c, h, w = input_image.shape
    attention_map = F.interpolate(attention_map, (h, w), mode='bilinear')
    threshold = random.uniform(0.25, 0.75)
    ret = input_image * (attention_map > threshold)
    return ret


def ranking_loss(coarse, fine, label): # binary
    b = label.shape[0]
    coarse = torch.softmax(coarse, dim=1)
    fine = torch.softmax(fine, dim=1)
    distance = 0
    for step, i in enumerate(label):
        if i == 0:
            distance += (coarse[step, i] - torch.sum(fine[step, 0:2]) + 0.05).clamp(min=0) / b
        elif i == 1:
            distance += (coarse[step, i] - torch.sum(fine[step, 2:5]) + 0.05).clamp(min=0) / b
    return distance

def train(train_loader,
          net1,
          optimizer1,
          net2, 
          optimizer2,
          epoch,
          normed_weight,
          writer=None,
          batch_size=16,
          total_epoch=100,
          save_path="ROI_ResNet50_0906",
          grad_norm=10.0):
    
    normed_weight_coarse, normed_weight_fine = normed_weight
    
    loss_record = AvgMeter()
    loss_crop_record = AvgMeter()
    loss_rank_record = AvgMeter()
    total_step = len(train_loader)
    for step, pack in enumerate(train_loader, start=1):
        net1.train()
        net2.train()
        # ---- data prepare ----
        images, labels = pack
        images = Variable(images).cuda()
        
        labels_coarse = 1 * (labels >= 2)
        labels_coarse = Variable(labels_coarse).long().cuda()
        
        labels_fine = Variable(labels).long().cuda()

        # ---- forward ----
        r1 = net1(images)

        # ---- calculate loss ----
        loss = F.cross_entropy(input=r1, target=labels_coarse, weight=normed_weight_coarse,)

        # ---- loss backward ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net1.parameters(), grad_norm)
        optimizer1.step()
        optimizer1.zero_grad()
        
        # ---- record ----
        loss_record.update(loss.data, batch_size)

        # ---- train visualization ----
        if step % 10 == 0 or step == total_step:
            print("------ {} ------".format(datetime.now()))
            print("------ Net1 ------")
            print("Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [CE_Loss: {:.4f}]".
                  format(epoch, total_epoch, step, total_step, loss_record.show()))
            
        # ---- ret training ----
        
        cam_extractor = CAM(net1, target_layer='classifier.7', fc_layer='classifier.9')
        out = net1(images.detach())
        cam_benign = cam_extractor(0, out,)[0].unsqueeze(1)
        cam_malignant = cam_extractor(1, out,)[0].unsqueeze(1)
        if torch.isnan(cam_benign).any() or torch.isnan(cam_malignant).any():
            continue
            
        images_crop = attention_crop(images, cam_malignant)
        r2 = net2(torch.cat([images_crop, images], dim=0))

        # ---- calculate loss ----
        ce_loss = F.cross_entropy(input=r2, target=torch.cat([labels_fine, labels_fine], dim=0), weight=normed_weight_fine,)
        rank_loss = ranking_loss(r1.detach(), r2, labels_coarse)
        loss = 1.0 * ce_loss + 5.0 * rank_loss

        # ---- loss backward ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net2.parameters(), grad_norm)
        optimizer2.step()
        optimizer2.zero_grad()
        
        # ---- record ----
        loss_crop_record.update(ce_loss.data, batch_size)
        loss_rank_record.update(rank_loss.data, batch_size)

        if step % 10 == 0 or step == total_step:
            print("------ Net2 ------")
            print("Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [CE_Loss: {:.4f}, Rank_Loss: {:.4f}]".
                  format(epoch, total_epoch, step, total_step, loss_crop_record.show(), loss_rank_record.show()))

    if writer is not None:
        writer.add_scalar('Training/Loss', loss_crop_record.show(), epoch)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path, exist_ok=True)
    
    if epoch % 5 == 0:
        torch.save(net1.state_dict(), save_path + 'net1_epoch_%d.pth' % epoch)
        torch.save(net2.state_dict(), save_path + 'net2_epoch_%d.pth' % epoch)
    return True


def val(net1,
        net2,
        epoch,
        best_acc,
        val_loader,
        normed_weight,
        writer=None,
        save_flag=False,
        save_path='ROI_ResNet50_0906',
        val_flag='val'):
    net1.eval()
    net2.eval()
    
    normed_weight_coarse, normed_weight_fine = normed_weight
    loss_net1_bank = []
    acc_net1_bank = []
    loss_net2_bank = []
    acc_net2_bank = []
    
    for step, pack in enumerate(val_loader, start=1):
        # ---- data prepare ----
        image, label = pack
        image = image.cuda()
        
        label_coarse = 1 * (label >= 2)
        label_coarse = label_coarse.long().cuda()
        
        label_fine = label.long().cuda()
        
        label = label.long().cuda()

        # ---- forward ----     
        with torch.no_grad():
            r = net1(image)
        # ---- calculate loss ----
        loss = F.cross_entropy(input=r, target=label_coarse, weight=normed_weight_coarse,)
        loss = loss.item()
        cls_acc = overall_acc(r, label_coarse)

        loss_net1_bank.append(loss)
        acc_net1_bank.append(cls_acc.cpu().numpy())
        
        # cam_extractor = CAM(net1, target_layer='classifier.7', fc_layer='classifier.9')
        # out = net1(image.detach())
        # cam_benign = cam_extractor(0, out,)[0].unsqueeze(1)
        # cam_malignant = cam_extractor(1, out,)[0].unsqueeze(1)

        # image = attention_drop(image, cam_benign)
        with torch.no_grad():
            r = net2(image)
            
        loss = F.cross_entropy(input=r, target=label_fine, weight=normed_weight_fine,)
        loss = loss.item()
        cls_acc = overall_acc(r, label_fine)

        loss_net2_bank.append(loss)
        acc_net2_bank.append(cls_acc.cpu().numpy())

    print('Net1 {} Loss: {:.4f},  Acc: {:.4f}'.
          format(val_flag, np.mean(loss_net1_bank), np.mean(acc_net1_bank)))
    print('Net2 {} Loss: {:.4f},  Acc: {:.4f}'.
          format(val_flag, np.mean(loss_net2_bank), np.mean(acc_net2_bank)))

    if writer is not None:
        writer.add_scalar('Validation/Loss', np.mean(loss_net2_bank), epoch)
        writer.add_scalar('Validation/Acc', np.mean(acc_net2_bank), epoch)
        
    if save_flag and best_acc < np.mean(acc_net2_bank):
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        torch.save(net1.state_dict(), save_path + 'net1_BestAcc.pth')
        torch.save(net2.state_dict(), save_path + 'net2_BestAcc.pth')
    return np.mean(acc_net2_bank)


if __name__ == "__main__":
    """
    需要修改的参数:
    model:        模型选择
    save_name:    保存模型名称
    save_path:    保存路径
    others:       学习率，batch_size，训练轮数等
    如果需要更改数据集需要在dataset里另行处理
    """
    num_classes = 5
    net1 = CoarseNet()
    net1 = net1.cuda()
    print("net1 (coarsenet) has {} paramerters in total".format(sum(x.numel() for x in net1.parameters())))
    net2 = FineNet()
    net2 = net2.cuda()
    print("net2 (finenet) has {} paramerters in total".format(sum(x.numel() for x in net2.parameters())))
    
    total_epoch = 50
    batch_size = 16
    seed = 108
    grad_norm = 10.0
    base_lr = 1e-4
    image_size = (352, 352)
    save_name = 'ROI_404_SEResNext50_5cls_20240724'
    save_path = '/data2/ceiling/workspace/RCFNet/save_model/{}/'.format(save_name)
    writer = SummaryWriter(save_path + 'logs')
    json_path = '/data1/ceiling/workspace/Gross/RCFNet/data_split/ROI_404_standard.json'
    root_path = '/data3/ceiling/datasets/Gross-Combined/ROI'
    
    optimizer1 = torch.optim.AdamW(net1.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-4)
    optimizer2 = torch.optim.AdamW(net2.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-4)

    label_map = {
    'AIS' : 0,
    'MIA' : 1,
    '1'   : 2,
    '2'   : 3,
    '3'   : 4,
    }
    
    train_dataset = GrossDataset(root_path = root_path,
                                 json_path = json_path,
                                 mode = 'train',
                                 label_map = label_map,
                                 image_size = image_size)
    label_count = []
    for i in range(num_classes):
        value = np.sum(np.array(train_dataset.label_list) == i)
        label_count.append(value)
        
    print("label count:", label_count)
    loss_weight_fine = []
    for i in label_count:
        loss_weight_fine.append(1.0 / i)
    normed_weight_fine = []
    for i in loss_weight_fine:
        normed_weight_fine.append(i / sum(loss_weight_fine))
    print("normed_weight_fine:", normed_weight_fine)
    normed_weight_fine = torch.Tensor(normed_weight_fine).cuda()
    
    loss_weight_coarse = [1.0 / (label_count[0] + label_count[1]), 1.0 / (label_count[2] + label_count[3] + label_count[4])]
    normed_weight_coarse = []
    for i in loss_weight_coarse:
        normed_weight_coarse.append(i / sum(loss_weight_coarse))
    print("normed_weight_coarse:", normed_weight_coarse)
    normed_weight_coarse = torch.Tensor(normed_weight_coarse).cuda()
    normed_weight = [normed_weight_coarse, normed_weight_fine]
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)

    val_dataset = ValDataset(root_path = root_path,
                             json_path = json_path,
                             mode = 'val',
                             label_map = label_map,
                             image_size = image_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                             num_workers=4, pin_memory=True)
    
    total_step = len(train_loader)
    best_acc = 0.0

    print("*" * 20, "|| Start Training ||", "*" * 20)
    for epoch in range(1, total_epoch + 1):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        train(train_loader,
              net1,
              optimizer1,
              net2,
              optimizer2,
              epoch,
              normed_weight,
              writer=writer,
              batch_size=batch_size,
              total_epoch=total_epoch,
              save_path=save_path,
              grad_norm=grad_norm,)
        val_acc = val(net1,
                      net2,
                      epoch,
                      best_acc,
                      val_loader,
                      normed_weight,
                      writer=writer,
                      save_flag=True,
                      save_path=save_path)
        if val_acc > best_acc:
            best_acc = val_acc
            print("best acc. is {:.4f}".format(val_acc))
    writer.close()