import torch
from torch.autograd import Variable
import torch.nn.functional as F
import timm
import numpy as np
from datetime import datetime
import os
from utils import AvgMeter, overall_acc
from dataset_with_clinic import GrossDataset, ValDataset
from torch.utils.tensorboard import SummaryWriter
from models import CoarseNet, FineNet, Fusion_clinic
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


def attention_drop(input_image, attention_map):
    b, c, h, w = input_image.shape
    attention_map = F.interpolate(attention_map, (h, w), mode='bilinear')
    ret = input_image * (attention_map > 0.50)
    return ret

def consistent_loss(coarse, fine):
    coarse = torch.softmax(coarse, dim=-1)
    fine = torch.softmax(fine, dim=-1)
    refine = torch.cat([torch.sum(fine[:, 0:2], dim=-1, keepdim=True), torch.sum(fine[:, 2:5], dim=-1, keepdim=True)], dim=-1)
    cosine = torch.sum(coarse * fine, dim=-1, keepdim=True) / (torch.sum(torch.sqrt(coarse ** 2), dim=-1, keepdim=True) * torch.sum(torch.sqrt(fine ** 2), dim=-1, keepdim=True))
    loss = torch.mean(1 - cosine)
    return loss
    
def train(train_loader,
          net1,
          net2, 
          model,
          optimizer,
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
    total_step = len(train_loader)
    for step, pack in enumerate(train_loader, start=1):
        model.train()
        # ---- data prepare ----
        images, clinics, labels = pack

        images = Variable(images).cuda()
        clinics = Variable(clinics).cuda()
        
        labels_coarse = 1 * (labels >= 2)
        labels_coarse = Variable(labels_coarse).long().cuda()
        
        labels_fine = Variable(labels).long().cuda()
        
        # ---- forward ----
        r1 = net1.classifier[:-1](images)
        cam_extractor = CAM(net1, target_layer='classifier.7', fc_layer='classifier.9')
        out = net1(images.detach())
        # cam_malignant = cam_extractor(1, out,)[0].unsqueeze(1)
        # images = attention_crop(images, cam_malignant)
        r2 = net2.classifier[:-1](images)

        c, f = model(torch.cat([r1, r2], dim=1), clinics)
        # ---- calculate loss ----
        loss = F.cross_entropy(input=c, target=labels_coarse, weight=normed_weight_coarse,) + F.cross_entropy(input=f, target=labels_fine, weight=normed_weight_fine,) 

        # ---- loss backward ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net1.parameters(), grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        
        # ---- record ----
        loss_record.update(loss.data, batch_size)

        # ---- train visualization ----
        if step % 10 == 0 or step == total_step:
            print("------ {} ------".format(datetime.now()))
            print("Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [CE_Loss: {:.4f}]".
                  format(epoch, total_epoch, step, total_step, loss_record.show()))

    if writer is not None:
        writer.add_scalar('Training/Loss', loss_record.show(), epoch)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path, exist_ok=True)
    
    if epoch % 1 == 0:
        torch.save(model.state_dict(), save_path + 'fusion_epoch_%d.pth' % epoch)
    return True


def val(net1,
        net2,
        model,
        epoch,
        best_acc,
        val_loader,
        normed_weight,
        writer=None,
        save_flag=False,
        save_path='ROI_ResNet50_0906',
        val_flag='val'):
    normed_weight_coarse, normed_weight_fine = normed_weight

    model.eval()
    loss_bank = []
    acc_coarse_bank = []
    acc_fine_bank = []
    
    for step, pack in enumerate(val_loader, start=1):
        # ---- data prepare ----
        image, clinic, label = pack
        image = image.cuda()
        clinic = clinic.cuda()
        label_coarse = 1 * (label >= 2)
        label_coarse = label_coarse.long().cuda()
        
        label_fine = label.long().cuda()
        
        label = label.long().cuda()
        r1 = net1.classifier[:-1](image)
        # cam_extractor = CAM(net1, target_layer='classifier.7', fc_layer='classifier.9')
        # out = net1(image.detach())
        # cam_malignant = cam_extractor(1, out,)[0].unsqueeze(1)
        # image = attention_crop(image, cam_malignant)
        r2 = net2.classifier[:-1](image)
        # ---- forward ----     
        with torch.no_grad():
            c, f = model(torch.cat([r1, r2], dim=-1), clinic)
        # ---- calculate loss ----
        loss = 0.2 * F.cross_entropy(input=c, target=label_coarse, weight=normed_weight_coarse,) + 0.8 * F.cross_entropy(input=f, target=label_fine, weight=normed_weight_fine,)
            
        loss = loss.item()
        loss_bank.append(loss)
        
        cls_acc = overall_acc(c, label_coarse)
        acc_coarse_bank.append(cls_acc.cpu().numpy())
        
        cls_acc = overall_acc(f, label_fine)
        acc_fine_bank.append(cls_acc.cpu().numpy())

    print('Fusion {} Loss: {:.4f},  Coarse_Acc: {:.4f}, Fine_Acc: {:.4f}'.
          format(val_flag, np.mean(loss_bank), np.mean(acc_coarse_bank), np.mean(acc_fine_bank)))

    if writer is not None:
        writer.add_scalar('Validation/Loss', np.mean(loss_bank), epoch)
        writer.add_scalar('Validation/Acc', np.mean(acc_fine_bank), epoch)
        
    if save_flag and best_acc < np.mean(acc_fine_bank):
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), save_path + 'Fusion_BestAcc.pth')
    return np.mean(acc_fine_bank)


if __name__ == "__main__":
    """
    需要修改的参数:
    model:        模型选择
    save_name:    保存模型名称
    save_path:    保存路径
    others:       学习率，batch_size，训练轮数等
    如果需要更改数据集需要在dataset里另行处理
    """
    
    weight1_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_404_SEResNext50_5cls_20240724/net1_epoch_20.pth'
    weight2_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_404_SEResNext50_5cls_20240724/net2_epoch_20.pth'
    
    num_classes = 5
    net1 = CoarseNet()
    net1 = net1.cuda()
    net1.load_state_dict(torch.load(weight1_path))
    net1.eval()
    print("net1 has {} paramerters in total".format(sum(x.numel() for x in net1.parameters())))
    
    net2 = FineNet()
    net2 = net2.cuda()
    net2.load_state_dict(torch.load(weight2_path))
    net2.eval()
    print("net2 has {} paramerters in total".format(sum(x.numel() for x in net2.parameters())))
    
    model = Fusion_clinic(in_channels=4096, num_clinic=9, num_features=256)
    model = model.cuda()
    print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    
    total_epoch = 10
    batch_size = 16
    seed = 108
    grad_norm = 10.0
    base_lr = 5e-5
    image_size = (352, 352)
    save_name = 'ROI_404_Clinic_Fusion_20240724'
    save_path = '/data2/ceiling/workspace/RCFNet/save_model/{}/'.format(save_name)
    writer = SummaryWriter(save_path + 'logs')
    json_path = '/data1/ceiling/workspace/Gross/RCFNet/data_split/ROI_404_standard.json'
    root_path = '/data3/ceiling/datasets/Gross-Combined/ROI'
    clinic_path = '/data1/ceiling/workspace/Gross/RCFNet/excels/GDPH_image_clinic2024_model.csv'
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=5e-3)

    label_map = {
    'AIS' : 0,
    'MIA' : 1,
    '1'   : 2,
    '2'   : 3,
    '3'   : 4,
    }
    
    train_dataset = GrossDataset(root_path = root_path,
                                 json_path = json_path,
                                 clinic_path = clinic_path,
                                 mode = 'train',
                                 label_map = label_map,
                                 image_size = image_size,
                                 clinic_info = ['age', 'sex', 'weight', 'height', 'BMI', 'smooking', 'smooking frequency', 'smooking history', 'maximum diameter'])
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
                             clinic_path = clinic_path,
                             mode = 'val',
                             label_map = label_map,
                             image_size = image_size,
                             clinic_info = ['age', 'sex', 'weight', 'height', 'BMI', 'smooking', 'smooking frequency', 'smooking history', 'maximum diameter'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                             num_workers=4, pin_memory=True)
    
    total_step = len(train_loader)
    best_acc = 0.0

    print("*" * 20, "|| Start Training ||", "*" * 20)
    for epoch in range(1, total_epoch + 1):
        optimizer.zero_grad()
        train(train_loader,
              net1,
              net2,
              model,
              optimizer,
              epoch,
              normed_weight,
              writer=writer,
              batch_size=batch_size,
              total_epoch=total_epoch,
              save_path=save_path,
              grad_norm=grad_norm,)
        val_acc = val(net1,
                      net2,
                      model,
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