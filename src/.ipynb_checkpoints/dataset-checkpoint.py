import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from constant import *

class XrayDataset(Dataset):
    '''
    Get image for train, validate and test base on NIH split
    '''
    
    def __init__(self, 
                 image_list_file=CHEXNET_TEST_CSV, 
                 transform=None, 
                 percentage=PERCENTAGE):
        data = pd.read_csv(image_list_file, sep=' ', header=None)
        self.images = data.iloc[:, 0].as_matrix()
        self.labels = data.iloc[:, 1:].as_matrix()
        self.transform = transform
        self.percentage = percentage
    
    def __getitem__(self, index):
        image_file = DATA_DIR + self.images[index]
        image = Image.open(image_file).convert('RGB')
#         image = imread(image_file)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(label)
    
    def __len__(self):
        return int(self.images.shape[0] * self.percentage)