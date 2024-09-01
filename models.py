from __future__ import print_function, division

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch import einsum
import torchvision
import timm
from einops import rearrange
from timm.models.resnet import Bottleneck, downsample_conv

import torchcam
from torchcam.methods import CAM, GradCAM


class CoarseNet(nn.Module):
    '''
    Coase Network is used for two-class classification
    '''
    def __init__(self, num_classes = 2): # benign, malignant
        super().__init__()
        backbone = timm.create_model('seresnext50_32x4d', pretrained=True, num_classes=num_classes, 
                                     pretrained_cfg_overlay=dict(file='./pretrained_weights/seresnext50_32x4d.bin'))
        self.classifier = nn.Sequential(backbone.conv1,
                                        backbone.bn1,
                                        backbone.act1,
                                        backbone.maxpool,
                                        backbone.layer1,
                                        backbone.layer2,
                                        backbone.layer3,
                                        backbone.layer4,
                                        backbone.global_pool,
                                        backbone.fc)
    
    def forward(self, x):
        x = self.classifier(x)
        return x

    
class FineNet(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()
        backbone = timm.create_model('seresnext50_32x4d', pretrained=True, num_classes=num_classes, 
                                     pretrained_cfg_overlay=dict(file='./pretrained_weights/seresnext50_32x4d.bin'))
        self.classifier = torch.nn.Sequential(backbone.conv1,
                                           backbone.bn1,
                                           backbone.act1,
                                           backbone.maxpool,
                                           backbone.layer1,
                                           backbone.layer2,
                                           backbone.layer3,
                                           backbone.layer4,
                                           backbone.global_pool,
                                           backbone.fc)
    
    def forward(self, x):
        x = self.classifier(x)
        return x


class Fusion(nn.Module):
    def __init__(self, in_channels=4096):
        super().__init__()
        self.coarse_layer = nn.Linear(in_channels, 2, bias=False)
        self.fine_layer = nn.Linear(in_channels, 5, bias=False)
        
    def forward(self, x):
        c = self.coarse_layer(x)
        f = self.fine_layer(x)
        return c, f



class Post(nn.Module):
    def __init__(self, 
                 weight1_path='/data1/ceiling/workspace/gross_models/save_model/ROI_AttentionCropFine_1010/net1_BestAcc.pth',
                 weight2_path='/data1/ceiling/workspace/gross_models/save_model/ROI_AttentionCropFine_1010/net2_BestAcc.pth',
                 weight3_path='/data1/ceiling/workspace/gross_models/save_model/ROI_AttentionCropFine_1010/Fusion_BestAcc.pth'):
        super().__init__()
        self.net1 = CoarseNet()
        self.net1.load_state_dict(torch.load(weight1_path))
        self.net2 = FineNet()
        self.net2.load_state_dict(torch.load(weight2_path))
        self.fusion = Fusion()
        self.fusion.load_state_dict(torch.load(weight3_path))
    
    def forward(self, x):
        r1 = self.net1.classifier[:-1](x)
        r2 = self.net2.classifier[:-1](x)
        c, f = self.fusion(torch.cat([r1, r2], dim=-1))
        return c, f

    
class Fusion_clinic(nn.Module):
    def __init__(self, in_channels=4096, num_clinic=9, num_features=256):
        super().__init__()
        self.coarse_layer = nn.Linear(in_channels + num_clinic * num_features, 2, bias=False)
        self.fine_layer = nn.Linear(in_channels + num_clinic * num_features, 5, bias=False)
        self.modules1 = nn.ModuleList()
        self.modules2 = nn.ModuleList()
        for i in range(num_clinic):
            self.modules1.append(self.clinic_layers(1, 256))
            self.modules2.append(self.clinic_layers(256, 256))
    
    def clinic_layers(self,
                      in_channels, 
                      out_channels):
        return nn.Sequential(nn.Linear(in_channels, out_channels),
                             nn.BatchNorm1d(out_channels),
                             nn.ReLU())
    
    def forward(self, x, clinic):
        features = [x]
        for i in range(clinic.shape[1]):
            out = self.modules1[i](clinic[:, i:i+1])
            out = self.modules2[i](out)
            features.append(out)
        x = torch.concat(features, dim=1)
        c = self.coarse_layer(x)
        f = self.fine_layer(x)
        return c, f
    
    
class Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super(Baseline, self).__init__()
        
        backbone = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=num_classes, 
                                     pretrained_cfg_overlay=dict(file='./pretrained_weights/resnext50_32x4d.bin'))
        
        self.classifier = nn.Sequential(backbone.conv1,
                                        backbone.bn1,
                                        backbone.act1,
                                        backbone.maxpool,
                                        backbone.layer1,
                                        backbone.layer2,
                                        backbone.layer3,
                                        backbone.layer4,
                                        backbone.global_pool, 
                                        backbone.fc)
        
    def forward(self, x):
        x = self.classifier(x)
        return x


class RCF(nn.Module):
    def __init__(self, 
                 image_scale=352,
                 num_classes=2):
        super(RCF, self).__init__()
        self.net = Baseline(num_classes = num_classes)
            
    def generate_cam(self, x, c=0):
        x = x.detach()
        cam_extractor = CAM(self.net, target_layer='classifier.7', fc_layer='classifier.9')
        out = self.classifier(x)
        cam = cam_extractor(c, out,)[0].unsqueeze(1)
        return cam
        
    def forward(self, x):
        x = self.net(x)
        return x