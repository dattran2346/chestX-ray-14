import torchvision
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F
from collections import OrderedDict
from constant import SCALE_FACTOR
import math

class Resnet(nn.Module):
    
    def __init__(self, variant):
        super(Resnet, self).__init__()
        assert variant in ['resnet50', 'resnet101', 'resnet152']
        
        # load retrain model
        model = pretrainedmodels.__dict__[variant](num_classes=1000, pretrained='imagenet')
        self.features = nn.Sequential(OrderedDict([
            ('conv1', model.conv1),
            ('bn1', model.bn1),
            ('relu', model.relu),
            ('maxpool', model.maxpool),
            ('layer1', model.layer1),
            ('layer2', model.layer2),
            ('layer3', model.layer3),
            ('layer4', model.layer4)
        ]))
        num_ftrs = model.last_linear.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )
        
        # load other info
        # load other info
        self.mean = model.mean
        self.std = model.std
        self.input_size = model.input_size[1] # assume every input is a square image
        self.input_range = model.input_range
        self.input_space = model.input_space
        self.resize_size = int(math.floor(self.input_size / SCALE_FACTOR))
        
    def forward(self, x):
        x = self.features(x) # 1x2048x7x7
        s = x.size()[3] # 7 if input image is 224x224, 16 if input image is 512x512
        x = F.avg_pool2d(x, kernel_size=s, stride=1) # 1x2048x1x1
        x = x.view(x.size(0), -1) # 1x2048
        x = self.classifier(x) # 1x1000
        return x
        
    def extract(self, x):
        return self.features(x)
    
def build(variant):
    net = Resnet(variant).cuda()
    return net

architect='resnet'