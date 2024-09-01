import os
import random
import json

import numpy as np
import imageio
import cv2
from PIL import Image

import torch
from torch import nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import albumentations as A
import pandas as pd

label_map = {
    'AIS' : 0,
    'MIA' : 1,
    '1'   : 2,  # 贴壁生长型
    '2'   : 3,  # 腺泡和乳头状
    '3'   : 4,  # 微乳头和实性
}


class GrossDataset(data.Dataset):
    """
    The Dataset of Gross Images
    """
    def __init__(self,
                 root_path,
                 json_path,
                 clinic_path,
                 label_map,
                 clinic_info = ['age', 'ct maximum diameter'],
                 mode='train',
                 image_size=(352, 352)):
        """
        params:
            root_path : saves the image and corresponding label dir
            json_path : data split strategy
            label_map : dir name to label
            image_size (h, w) : the size of image
        """
        super(GrossDataset, self).__init__()
        
        self.image_list = []
        self.label_list = []
        self.clinic_list = []
        
        with open(json_path, 'r') as f:
            patient = json.load(f)
        patient = patient[mode]
        df_clinic = pd.read_csv(clinic_path)
        
        for label in os.listdir(root_path):
            if label[0] == '.': # nothing
                continue
            else:
                path = os.path.join(root_path, label)
                for name in os.listdir(path):
                    if name[0] == '.':
                        continue
                    elif name[:4] in patient[label]:
                        self.image_list.append(os.path.join(path, name))
                        self.label_list.append(label_map[label])
                        row = df_clinic.loc[df_clinic['number'] == int(name[:4])]
                        clinic = []
                        for p in clinic_info:
                            clinic += row[p].to_list()
                        self.clinic_list.append(clinic)
                        
                        
        self.transform = A.Compose([
            A.Resize(400, 400, always_apply=True),
            A.RandomCrop(image_size[0], image_size[1], always_apply=True),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=20, p=0.5),
        ])
        
        self.img_transfom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = self.rgb_loader(self.image_list[index])
        img = np.array(img)
        transformed = self.transform(image=img)
        img = self.img_transfom(transformed['image'])
        clinic = np.array(self.clinic_list[index], dtype=np.float32)
        clinic = torch.from_numpy(clinic)
        label = np.array(self.label_list[index], dtype=np.float32)
        label = torch.from_numpy(label)
        return img, clinic, label


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
        
class ValDataset(data.Dataset):
    """
    The Validation Dataset of Gross Images
    """
    def __init__(self,
                 root_path,
                 json_path,
                 clinic_path,
                 label_map,
                 clinic_info = ['age', 'ct maximum diameter'],
                 mode='val',
                 image_size=(352, 352)):
        """
        params:
            root_path : saves the image and corresponding label dir
            json_path : data split strategy
            label_map : dir name to label
            image_size (h, w) : the size of image
        """
        super(ValDataset, self).__init__()
        
        self.image_list = []
        self.label_list = []
        self.clinic_list = []
        df_clinic = pd.read_csv(clinic_path)
        
        with open(json_path, 'r') as f:
            patient = json.load(f)
        patient = patient[mode]
        df_clinic = pd.read_csv(clinic_path)
        
        for label in os.listdir(root_path):
            if label[0] == '.': # nothing
                continue
            else:
                path = os.path.join(root_path, label)
                for name in os.listdir(path):
                    if name[0] == '.':
                        continue
                    elif name[:4] in patient[label]:
                        self.image_list.append(os.path.join(path, name))
                        self.label_list.append(label_map[label])
                        row = df_clinic.loc[df_clinic['number'] == int(name[:4])]
                        clinic = []
                        for p in clinic_info:
                            clinic += row[p].to_list()
                        self.clinic_list.append(clinic)
                        
        self.transform = A.Compose([
            A.Resize(image_size[0], image_size[1], always_apply=True),
        ])
        
        self.img_transfom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = self.rgb_loader(self.image_list[index])
        img = np.array(img)
        transformed = self.transform(image=img)
        img = self.img_transfom(transformed['image'])
        clinic = np.array(self.clinic_list[index], dtype=np.float32)
        clinic = torch.from_numpy(clinic)
        label = np.array(self.label_list[index], dtype=np.float32)
        label = torch.from_numpy(label)
        return img, clinic, label

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
        
        
class TestDataset(data.Dataset):
    """
    The Test Dataset of Gross Images
    """
    def __init__(self,
                 root_path,
                 json_path,
                 clinic_path,
                 label_map,
                 clinic_info = ['age', 'ct maximum diameter'],
                 mode='test',
                 name_length=4,
                 image_size=(352, 352)):
        """
        params:
            root_path : saves the image and corresponding label dir
            json_path : data split strategy
            label_map : dir name to label
            image_size (h, w) : the size of image
        """
        super(TestDataset, self).__init__()
        
        self.image_list = []
        self.label_list = []
        self.name_list = []
        self.clinic_list = []
        
        with open(json_path, 'r') as f:
            patient = json.load(f)
        patient = patient[mode]
        df_clinic = pd.read_csv(clinic_path)
        
        for label in os.listdir(root_path):
            if label[0] == '.': # nothing
                continue
            else:
                path = os.path.join(root_path, label)
                for name in os.listdir(path):
                    if name[0] == '.':
                        continue
                    elif name[:name_length] in patient[label]:
                        
                        # 名字的形式
                        row = df_clinic.loc[df_clinic['number'] == int(name[:name_length])]
                        # row = df_clinic.loc[df_clinic['number'] == name[:name_length]]
                        clinic = []
                        for p in clinic_info:
                            clinic += row[p].to_list()
                        if len(clinic) == 0:
                            # print(name)
                            continue
                        self.clinic_list.append(clinic)
                        self.image_list.append(os.path.join(path, name))
                        self.label_list.append(label_map[label])
                        self.name_list.append(name)
                        
        self.transform = A.Compose([
            A.Resize(image_size[0], image_size[1], always_apply=True),
        ])
        
        self.img_transfom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = self.rgb_loader(self.image_list[index])
        img = np.array(img)
        label = np.array(self.label_list[index], dtype=np.float32)
        label = torch.from_numpy(label)
        transformed = self.transform(image=img)
        img = self.img_transfom(transformed['image'])
        name = self.name_list[index]
        clinic = np.array(self.clinic_list[index], dtype=np.float32)
        clinic = torch.from_numpy(clinic)
        return img, clinic, label, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')