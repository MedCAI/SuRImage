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


def train(train_loader,
          model,
          optimizer,
          epoch,
          normed_weight,
          writer=None,
          batch_size=16,
          total_epoch=100,
          save_path="ROI_ResNet50_0906",
          grad_norm=10.0):
    
    model.train()
    loss_record = AvgMeter()
    total_step = len(train_loader)
    for step, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, labels = pack

        images = Variable(images).cuda()
        labels = Variable(labels).long().cuda()

        # ---- forward ----
        r = model(images)

        # ---- calculate loss ----
        loss = F.cross_entropy(input=r, target=labels, weight=normed_weight,)

        # ---- loss backward ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
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
    
    if epoch % 5 == 0:
        torch.save(model.state_dict(), save_path + 'epoch_%d.pth' % epoch)
    return True


def val(model,
        epoch,
        best_acc,
        val_loader,
        normed_weight,
        writer=None,
        save_flag=False,
        save_path='ROI_ResNet50_0906',
        val_flag='val'):
    model.eval()

    loss_bank = []
    acc_bank = []
    for step, pack in enumerate(val_loader, start=1):
        # ---- data prepare ----
        image, label = pack
        image = image.cuda()
        label = label.long().cuda()

        # ---- forward ----     
        with torch.no_grad():
            r = model(image)
        # ---- calculate loss ----
        loss = F.cross_entropy(input=r, target=label, weight=normed_weight,)
        loss = loss.item()

        cls_acc = overall_acc(r, label)

        loss_bank.append(loss)
        acc_bank.append(cls_acc.cpu().numpy())

    print('{} Loss: {:.4f},  Acc: {:.4f}'.
          format(val_flag, np.mean(loss_bank), np.mean(acc_bank)))

    if writer is not None:
        writer.add_scalar('Validation/Loss', np.mean(loss_bank), epoch)
        writer.add_scalar('Validation/Acc', np.mean(acc_bank), epoch)
        
    if save_flag and best_acc < np.mean(acc_bank):
        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), save_path + 'BestAcc.pth')
    return np.mean(acc_bank)


if __name__ == "__main__":
    """
    需要修改的参数:
    model:        模型选择
    save_name:    保存模型名称
    save_path:    保存路径
    others:       学习率，batch_size，训练轮数等
    如果需要更改数据集需要在dataset里另行处理
    """
    num_classes = 2
    model = timm.create_model('seresnext50_32x4d', pretrained=True, num_classes=num_classes, 
                              pretrained_cfg_overlay=dict(file='./pretrained_weights/seresnext50_32x4d.bin'))
    model = model.cuda()
    params = model.parameters()
    print("model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    total_epoch = 35
    batch_size = 32
    seed = 108
    grad_norm = 10.0
    base_lr = 1e-4
    image_size = (352, 352)
    save_name = 'ROI_404_SEResNext50_5cls_20240724'
    save_path = '/data2/ceiling/workspace/RCFNet/save_model/{}/'.format(save_name)
    writer = SummaryWriter(save_path + 'logs')
    json_path = '/data1/ceiling/workspace/Gross/RCFNet/data_split/ROI_404_standard.json'
    root_path = '/data3/ceiling/datasets/Gross-Combined/ROI/'
    
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), weight_decay=1e-4)

    label_map = {
    'AIS' : 0,
    'MIA' : 0,
    '1'   : 1,
    '2'   : 1,
    '3'   : 1,
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
    loss_weight = []
    for i in label_count:
        loss_weight.append(1.0 / i)
    normed_weight = []
    for i in loss_weight:
        normed_weight.append(i / sum(loss_weight))
    print("normed_weight:", normed_weight)
    normed_weight = torch.Tensor(normed_weight).cuda()
    
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
        optimizer.zero_grad()
        train(train_loader,
              model,
              optimizer,
              epoch,
              normed_weight,
              writer=writer,
              batch_size=batch_size,
              total_epoch=total_epoch,
              save_path=save_path,
              grad_norm=grad_norm,)
        val_acc = val(model,
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