from torch.utils.data import DataLoader, Dataset
from heatmap import HeatmapGenerator
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from constant import IMAGENET_MEAN, IMAGENET_STD
from fastai.core import to_np
from scipy import ndimage
from tqdm import tqdm


PATH = Path('/home/dattran/data/xray-thesis/chestX-ray14')
IMAGE_DN = 'images'
attention_dn = PATH/'tmp/attention'
attention_dn.mkdir(parents=True, exist_ok=True)

def crop_attention(heatmap, image_name):
    """
    heatmap: (7, 7) range [0, 1]
    image_name: 00007551_000.png, 00000001_000.png
    """
    image = Image.open(PATH/IMAGE_DN/image_name)

    heatmap = to_np(heatmap)
    heatmap = cv2.resize(heatmap, (1024, 1024))

    mask = heatmap > 0.7
    slice_y, slice_x = ndimage.find_objects(mask, True)[0]

    image = np.array(image)
    cropped = Image.fromarray(image[slice_y, slice_x])
    cropped.save(attention_dn/image_name)

class AttentionDataset(Dataset):
    def __init__(self):
        df = pd.read_csv('../csv/Data_Entry_Clean.csv')
        self.image_names = df['Image Index'].values
        self.tfm = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
                        ])

    def __getitem__(self, i):
        image = Image.open(PATH/IMAGE_DN/self.image_names[i]).convert('RGB')
        return self.tfm(image), self.image_names[i]

    def __len__(self):
        return len(self.image_names)

dataset = AttentionDataset()
dataloader = DataLoader(dataset, 16) # fastai dataloader
g = HeatmapGenerator()

for images, image_names in tqdm(iter(dataloader)):
    print()
    heatmaps, _ = g.generate(images)

    for i in range(16):
        heatmap = to_np(heatmaps[i])
        heatmap = cv2.resize(heatmap, (1024, 1024))
        crop_attention(heatmap, image_names[i])
