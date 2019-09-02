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
            normalize,
        ]),
        transforms.Compose([
            transforms.Resize((size, size)),
            toTensor,
            normalize,
        ]),
        # 'test': transforms.Compose([ # Use TTA using five crop
        #     transforms.Resize(resize_size),
        #     transforms.FiveCrop(size),
        #     transforms.Lambda(lambda crops: torch.stack([toTensor(crop) for crop in crops])),
        #     transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        # ])
    ]

def tta(resize=None, input_size=224):
    """
    case 1: resize to 256 and 5-crop
    case 2: transfrom segmented image from unet (size 256)
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    toTensor = transforms.ToTensor()
    tfms = []
    tfms.append(transforms.Resize(256))
    tfms.append(transforms.FiveCrop(input_size))
    # tfms.append(transforms.Resize(size))
    # tfms.append(toTensor)
    # tfms.append(normalize)
    tfms.append(transforms.Lambda(lambda crops: torch.stack([toTensor(crop) for crop in crops])))
    tfms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))

    return transforms.Compose(tfms)

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
