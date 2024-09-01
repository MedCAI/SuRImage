import PIL
import torchcam
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

from torchvision.models import resnet18
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
import timm
import torch
import imageio
import cv2
import matplotlib.pyplot as plt
import json
import os

from models import CoarseNet, FineNet, Fusion


if __name__ == "__main__":
    root_path = '/data5/ceiling/datasets/Gross-Combined/ROI'
    json_path = './data_split/ROI_combined_404.json'
    save_path = ''
    image_size = (352, 352)
    num_classes = 5
    mode = 'train'   # 'val'
    label_map = {'AIS' : 0,
                 'MIA' : 0,
                 '1'   : 1,  # 贴壁生长型
                 '2'   : 1,  # 腺泡和乳头状
                 '3'   : 1,  # 微乳头和实性
                }
    # label_list = ['AIS', 'MIA', '1级', '2级', '3级']
    label_list = ['non-IAC', 'IAC']

    weight_path = '/data2/ceiling/workspace/RCFNet/save_model/ROI_404_RCFNet_5cls_20240121/net1_epoch_35.pth'
    model = CoarseNet()
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    
    save_dir = '/data2/ceiling/workspace/RCFNet/attentionmaps/ROI_combined_404/coarse/train/'
    
    cam_extractor = GradCAMpp(model)
    
    with open(json_path, 'r') as f:
        patient = json.load(f)
    patient = patient[mode]
    
    flag = 0
    for label in os.listdir(root_path):
        if label[0] == '.': # nothing
            continue
        else:
            save_path = os.path.join(save_dir, label)
            if os.path.exists(save_path) is False:
                os.makedirs(save_path, exist_ok=True)
            path = os.path.join(root_path, label)
            for name in os.listdir(path):
                if name[0] == '.':
                    continue
                # if name[:4] == '0953':
                #    flag = 1
                # if flag == 0:
                #    continue
                elif name[:4] in patient[label]:
                    image_path = os.path.join(path, name)
                    img =  imageio.imread(image_path)
                    h, w, c = img.shape
                    if c == 4:
                        img = img[:, :, :3]
                    imageio.imwrite(os.path.join(save_path, name), img)
                    img = read_image(os.path.join(save_path, name))
                    if img.shape[0] == 4:
                        img = img[:3]
                    print(image_path, img.shape)
                    input_tensor = normalize(resize(img, image_size) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    out = model(input_tensor.unsqueeze(0))
                    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out, retain_graph=True)
                    img_name = name.split('.')[0] + '_' + label_list[out.squeeze(0).argmax().item()] + '.png'
                    cam = cv2.resize(activation_map[0].squeeze(0).detach().numpy(), image_size, interpolation=cv2.INTER_LINEAR)
                    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
                    imageio.imwrite(os.path.join(save_path, img_name), result)
                    
                    activation_map = cam_extractor(label_map[label], out, retain_graph=True)
                    img_name = name.split('.')[0] + '_GT_' + label_list[label_map[label]] + '.png'
                    cam = cv2.resize(activation_map[0].squeeze(0).detach().numpy(), image_size, interpolation=cv2.INTER_LINEAR)
                    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
                    imageio.imwrite(os.path.join(save_path, img_name), result)