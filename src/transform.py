from constant import *
from dataset import ChestXray14Dataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
# from pretrainedmodels.utils import ToRange255, ToSpaceBGR
from PIL import ImageOps
from constant import IMAGENET_MEAN, IMAGENET_STD


def chest_xray_transfrom(size, scale_factor):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    # toSpaceBGR = ToSpaceBGR(model.input_space=='BGR')
    # toRange255 = ToRange255(max(model.input_range)==255)
    toTensor = transforms.ToTensor()
    resize_size = int(math.floor(size / scale_factor))
    return [
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size),
            transforms.ColorJitter(0.15, 0.15),
            transforms.RandomRotation(15),
            toTensor,
            # toSpaceBGR,
            # toRange255,
            normalize,
        ]),
        transforms.Compose([
            transforms.Resize((size, size)),
            toTensor,
            # toSpaceBGR,
            # toRange255,
            normalize,
        ]),
        # 'test': transforms.Compose([ # Use TTA using five crop
        #     transforms.Resize(resize_size),
        #     transforms.FiveCrop(size),
        #     transforms.Lambda(lambda crops: torch.stack([toTensor(crop) for crop in crops])),
        #     transforms.Lambda(lambda crops: torch.stack([toSpaceBGR(crop) for crop in crops])),
        #     transforms.Lambda(lambda crops: torch.stack([toRange255(crop) for crop in crops])),
        #     transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        # ])
    ]


def lung_segmentation_transfrom(sz):
    return [
        (
            transforms.Compose([  # image
                transforms.Resize((sz, sz)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.15, 0.15),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]),
            transforms.Compose([  # mask
                transforms.Resize((sz, sz)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.15, 0.15),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
            ])
        ),
        (
            transforms.Compose([  # image
                transforms.Resize((sz, sz)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ]),
            transforms.Compose([  # mask
                transforms.Resize((sz, sz)),
                transforms.ToTensor(),
            ])
        )
    ]
