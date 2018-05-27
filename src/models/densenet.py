import torchvision
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F
import torch
from constant import SCALE_FACTOR
import math
import pdb

class DenseNet(nn.Module):
    
    def __init__(self, variant):
        super(DenseNet, self).__init__()
        assert variant in ['densenet121', 'densenet161', 'densenet201']
        
        # load retrain model 
        model = pretrainedmodels.__dict__[variant](num_classes=1000, pretrained='imagenet')
        self.features = model.features
        num_ftrs = model.last_linear.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )
        # TODO: BCELoss with logit for numeric stable
        # self.classifier = nn.Linear(num_ftrs, 14)
        
        # load other info
        self.mean = model.mean
        self.std = model.std
        self.input_size = model.input_size[1] # assume every input is a square image
        self.input_range = model.input_range
        self.input_space = model.input_space
        self.resize_size = int(math.floor(self.input_size / SCALE_FACTOR))
         
    def forward(self, x, kwargs):
        # pdb.set_trace()
        x = self.features(x) # 1x1024x7x7
        s = x.size()[3] # 7 if input image is 224x224, 16 if input image is 512x512
        x = F.relu(x, inplace=True) # 1x1024x7x7
        
        pooling = kwargs.pooling
        # pdb.set_trace()
        if pooling == 'MAX':
            x = F.max_pool2d(x, kernel_size=s, stride=1)
            x = x.view(x.size(0), -1) # 1x1024
        elif pooling == 'AVG':
            x = F.avg_pool2d(x, kernel_size=s, stride=1) # 1x1024x1x1
            x = x.view(x.size(0), -1) # 1x1024
        elif pooling == 'LSE':
            r = kwargs.lse_r
            x_max = F.max_pool2d(x, kernel_size=s, stride=1)
            p = ((1/r) * torch.log((1 / (s*s)) * torch.exp(r*(x - x_max)).sum(3).sum(2)))
            x_max = x_max.view(x.size(0), -1)
            x = x_max + p
        else:
            raise ValueError('Invalid pooling')
        
        x = self.classifier(x) # 1x1000
        return x
        
    def extract(self, x):
        return self.features(x)
    
    # def count_params(self):
    #     return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build(variant):
    net = DenseNet(variant).cuda()
    return net

architect='densenet'
    
    