from constant import *
from dataset import XrayDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

def train_dataloader(image_list_file='train_val_list.csv', percentage=PERCENTAGE):
    # TODO: Implement kFold for train test split
    normalize = transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_SD)
    transform = transforms.Compose([
        transforms.Resize(264),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=WIDTH),
        transforms.ColorJitter(0.15, 0.15),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])
    dataset = XrayDataset(image_list_file, transform, percentage)
    return DataLoader(dataset=dataset, batch_size=BATCHSIZE,
                      shuffle=True, num_workers=4, pin_memory=False)

def test_dataloader(image_list_file='test_list.csv', percentage=PERCENTAGE, agumented=TEST_AGUMENTED):
    normalize = transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_SD)
    if agumented:
        # base on https://github.com/arnoweng/CheXNet/blob/master/model.py
        transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(586),
            transforms.TenCrop(WIDTH),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(WIDTH),
            transforms.ToTensor(),
            normalize
        ])
    dataset = XrayDataset(image_list_file, transform, percentage)
    return DataLoader(dataset=dataset, batch_size=2*BATCHSIZE,
                      shuffle=False, num_workers=8, pin_memory=False)

def compute_aucs(targets, preds):
    aurocs = []
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()
    for i in range(N_CLASSES):
        aurocs.append(roc_auc_score(targets[:, i], preds[:, i]))
    return aurocs

#def plot_cam_results(crt)