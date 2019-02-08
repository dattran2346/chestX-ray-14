import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import random


class ChestXray14Dataset(Dataset):
    '''
    Get image for train, validate and test base on NIH split
    '''

    def __init__(self, image_names, labels, transform, path, size, percentage=0.1):
        self.labels = labels
        self.percentage = percentage
        self.size = size
        self.image_names = image_names
        self.path = path
        self.transform = transform

    def __getitem__(self, index):
        image_file = self.path/self.image_names[index]
        image = Image.open(image_file).convert('RGB') # 1 channel image
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return int(self.image_names.shape[0] * self.percentage)

    @property
    def sz(self):
        # fastai compatible: learn.summary()
        return self.size


class LungSegmentationDataset(Dataset):

    def __init__(self, image_names, mask_names, transform, path, size):
        self.image_names = image_names
        self.mask_names = mask_names
        self.path = path
        self.transform = transform
        self.size = size

    def __getitem__(self, i):
        image_file = self.path/'images'/self.image_names[i]
        mask_file = self.path/'masks'/self.mask_names[i]

        image = Image.open(image_file).convert('RGB')
        mask = Image.open(mask_file)

        seed = random.randint(0, 2**32)
        if self.transform:
            random.seed(seed)
            image = self.transform[0](image)
            random.seed(seed)
            mask = self.transform[1](mask)

        return image, mask

    def __len__(self):
        return len(self.image_names)

    @property
    def sz(self):
        return self.size
