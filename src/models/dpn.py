import torchvision
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F
from constant import SCALE_FACTOR
import math
from pretrainedmodels.models.dpn import adaptive_avgmax_pool2d

class DPN(nn.Module):
    
    def __init__(self, variant):
        super(DPN, self).__init__()
        assert variant in ['dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']
        
        # load retrain model 
        model = pretrainedmodels.__dict__[variant](num_classes=1000, pretrained='imagenet')
        self.features = model.features
        num_ftrs = model.classifier.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_ftrs, 14, kernel_size=1, bias=True), # something wrong here abt dimension
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
        x = self.features(x) # 1x1024x7x7
        if not self.training and self.test_time_tool:
            x = F.avg_pool2d(x, kernel_size=7, stride=1)
            x = self.classifier(x)
            x = adaptive_avgmax_pool2d(out, pool_type='avgmax') # something wrong here abt dimension
        else:
            x = adaptive_avgmax_pool2d(x, pool_type='avg')
            x = self.classifier(x)
        return x
        
    def extract(self, x):
        return self.features(x)

def build(variant):
    net = DPN(variant).cuda()
    return net

architect='dpn'
    
    