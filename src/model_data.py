from constant import *
from torch.utils.data import DataLoader
from dataset import ChestXray14Dataset, MontgomeryDataset
from transform import *
from pathlib import Path
from sklearn.model_selection import train_test_split
from fastai.dataset import ImageData


class ModelData(ImageData):
    def get_dl(self, ds, shuffle):
        # use torch DataLoader, instead of fastai DataLoader
        return DataLoader(ds, self.bs, shuffle=shuffle, num_workers=self.num_workers)

