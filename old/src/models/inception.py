import torchvision
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F
from collections import OrderedDict
from constant import SCALE_FACTOR
import math

class InceptionNet(nn.Module):
    
    def __init__(self, variant):
        super(InceptionNet, self).__init__()
        assert variant in ['inceptionv4', 'inceptionv3', 'inceptionresnetv2']
        
        # load pretrain model
        model = pretrainedmodels.__dict__[variant](num_classes=1000, pretrained='imagenet')
        self.features = _get_features(model, variant)
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
        x = self.features(x) # 1x1536x8x8
        s = x.size()[3] # 8 if input image is 224x224
        x = F.avg_pool2d(x, kernel_size=s, count_include_pad=False) # 1x1536x1x1, same for inceptionv4 and inceptionresnetv2
        x = x.view(x.size(0), -1) # 1x1536
        x = self.classifier(x) # 1x1000
        return x
        
    def extract(self, x):
        return self.features(x) # 1x1536x8x8
    
def build(variant):
    net = InceptionNet(variant).cuda()
    return net

def _get_features(model, variant):
    if variant == 'inceptionv4':
        features =  model.features
    elif variant == 'inceptionv3':
        # TODO: Take a look on this
        features = nn.Sequential(OrderedDict([
            ('Conv2d_1a_3x3', model.Conv2d_1a_3x3),
            ('Conv2d_2a_3x3', model.Conv2d_2a_3x3),
            ('Conv2d_2b_3x3', model.Conv2d_2b_3x3),
            ('max_pool2d_1', torch.nn.MaxPool2d(3, stride=2)),
            ('Conv2d_3b_1x1', model.Conv2d_3b_1x1),
            ('Conv2d_4a_3x3', model.Conv2d_4a_3x3),
            ('max_pool2d_2', torch.nn.MaxPool2d(3, stride=2)),
            ('Mixed_5b', model.Mixed_5b),
            ('Mixed_5c', model.Mixed_5c),
            ('Mixed_5d', model.Mixed_5d),
            ('Mixed_6a', model.Mixed_6a),
            ('Mixed_6b', model.Mixed_6b),
            ('Mixed_6c', model.Mixed_6c),
            ('Mixed_6d', model.Mixed_6b),
            # ('Mixed_6c', model.Mixed_6c),
        ]))
    elif variant == 'inceptionresnetv2':
        features = nn.Sequential(OrderedDict([
            ('conv2d_1a', model.conv2d_1a),
            ('conv2d_2a', model.conv2d_2a),
            ('conv2d_2b', model.conv2d_2b),
            ('maxpool_3a', model.maxpool_3a),
            ('conv2d_3b', model.conv2d_3b),
            ('conv2d_4a', model.conv2d_4a),
            ('maxpool_5a', model.maxpool_5a),
            ('mixed_5b', model.mixed_5b),
            ('repeat', model.repeat),
            ('mixed_6a', model.mixed_6a),
            ('repeat_1', model.repeat_1),
            ('mixed_7a', model.mixed_7a),
            ('repeat_2', model.repeat_2),
            ('block8', model.block8),
            ('conv2d_7b', model.conv2d_7b)
        ]))
    else:
        raise "Unknown variant"
    return features

architect='inception'