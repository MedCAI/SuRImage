from __future__ import print_function, division

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch import einsum
import torchvision
import timm


def rank_loss(coarse, fine):
    coarse = torch.softmax(coarse, dim=1)
    fine = torch.softmax(fine, dim=1)
    fine_lowrisk = torch.sum(fine[:, 0:2], dim=1, keepdim=True)
    fine_highrisk = torch.sum(fine[:, 2:5], dim=1, keepdim=True)
    coarse_lowrisk = coarse[:, 0:1]
    coarse_highrisk = coarse[:, 1:2]
    upper = (coarse_lowrisk) * (fine_lowrisk) + (coarse_highrisk) * (fine_highrisk)
    down = torch.sqrt((coarse_lowrisk) ** 2 + (coarse_highrisk) ** 2) * torch.sqrt((fine_lowrisk) ** 2 + (fine_highrisk) ** 2)
    return 1.0 - torch.mean(upper / (down + 1e-9))

