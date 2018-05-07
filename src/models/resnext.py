import torchvision
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F
from constant import SCALE_FACTOR
import math

class Resnext(nn.Module):
    
    def __init__(self, variant):
        super(Resnext, self).__init__()
        assert variant in ['resnext101_32x4d', 'resnext101_64x4d']
        
        # load retrain model 
        model = pretrainedmodels.__dict__[variant](num_classes=1000, pretrained='imagenet')
        self.features = model.features
        num_ftrs = model.last_linear.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )
        
        # load other info
        self.mean = model.mean
        self.std = model.std
        self.input_size = model.input_size[1] # assume every input is a square image
        self.input_range = model.input_range
        self.input_space = model.input_space
        self.resize_size = int(math.floor(self.input_size / SCALE_FACTOR))
         
    def forward(self, x):
        x = self.features(x) # 
        s = x.size()[3] # 7 if input image is 224x224, 16 if input image is 512x512
        x = F.avg_pool2d(x, kernel_size=(7, 7), stride=(1, 1)) # 1x1024x1x1
        x = x.view(x.size(0), -1) # 1x1024
        x = self.classifier(x) # 1x1000
        return x
        
    def extract(self, x):
        return self.features(x)

def build(variant):
    net = Resnext(variant).cuda()
    return net

architect='resnext'
    
    