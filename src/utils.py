from constant import *
from dataset import XrayDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import math
from pretrainedmodels.utils import ToRange255, ToSpaceBGR
from PIL import ImageOps

def train_dataloader(model, image_list_file=CHEXNET_TRAIN_CSV, percentage=PERCENTAGE, batch_size=BATCHSIZE):
    # TODO: Implement kFold for train test split
    tfs = []
    if PREPROCESS:
        tfs.append(transforms.Lambda(lambda image: ImageOps.equalize(image))) # equalize histogram 
    tfs.append(transforms.Resize(model.resize_size))
    tfs.append(transforms.RandomHorizontalFlip())
    tfs.append(transforms.RandomResizedCrop(size=model.input_size))
    tfs.append(transforms.ColorJitter(0.15, 0.15))
    tfs.append(transforms.RandomRotation(15))
    tfs.append(transforms.ToTensor())
    tfs.append(ToSpaceBGR(model.input_space=='RGB'))
    tfs.append(ToRange255(max(model.input_range)==255))
    tfs.append(transforms.Normalize(model.mean, model.std))
    transform = transforms.Compose(tfs)
    dataset = XrayDataset(image_list_file, transform, percentage)
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=True, num_workers=3, pin_memory=False)

# test_dataloader applied ten crop to validate dataset, but here we dont have sufficient memory to do so, so not use iot
def test_dataloader(model, image_list_file=CHEXNET_TEST_CSV, percentage=PERCENTAGE, batch_size=BATCHSIZE):
    normalize = transforms.Normalize(model.mean, model.std)
    toSpaceRGB = ToSpaceBGR(model.input_space=='RGB')
    toRange255 = ToRange255(max(model.input_range)==255)
    toTensor = transforms.ToTensor()
    tfs = []
    if PREPROCESS:
        tfs.append(transforms.Lambda(lambda image: ImageOps.equalize(image))) # equalize histogram 

    tfs.append(transforms.Resize(size=model.input_size))
    tfs.append(toTensor)
    tfs.append(toSpaceRGB)
    tfs.append(toRange255)
    tfs.append(normalize)
    
    transform =  transforms.Compose(tfs)
    dataset = XrayDataset(image_list_file, transform, percentage)
    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=False, num_workers=3, pin_memory=False)

def aucs(targets, preds):
    aurocs = []
    for i in range(N_CLASSES):
        aurocs.append(roc_auc_score(targets[:, i], preds[:, i]))
    return aurocs
