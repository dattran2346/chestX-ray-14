import torchvision
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F
from collections import OrderedDict

class Nasnet(nn.Module):
    
    def __init__(self, variant):
        super(Nasnet, self).__init__()
        assert variant in ['nasnetalarge']
        
        # load retrain model
        self.model = pretrainedmodels.__dict__[variant](num_classes=1000, pretrained='imagenet')
#         self.features = nn.Sequential(OrderedDict([
#             ('conv0', model.conv0),
#             ('cell_stem_0', model.cell_stem_0),
#             ('cell_stem_1', model.cell_stem_1),
#             ('cell_0', model.cell_0),
#             ('cell_1', model.cell_1),
#             ('cell_2', model.cell_2),
#             ('cell_3', model.cell_3),
#             ('cell_4', model.cell_4),
#             ('cell_5', model.cell_5),
#             ('reduction_cell_0', model.reduction_cell_0),
#             ('cell_6', model.cell_6),
#             ('cell_7', model.cell_7),
#             ('cell_8', model.cell_8),
#             ('cell_9', model.cell_9),
#             ('cell_10', model.cell_10),
#             ('cell_11', model.cell_11),
#             ('reduction_cell_1', model.reduction_cell_1),
#             ('cell_12', model.cell_6),
#             ('cell_13', model.cell_7),
#             ('cell_14', model.cell_8),
#             ('cell_15', model.cell_9),
#             ('cell_16', model.cell_10),
#             ('cell_17', model.cell_11)

#         ]))
        num_ftrs = self.model.last_linear.in_features
        self.model.last_linear = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )
        
        # load other info
        # load other info
        self.mean = self.model.mean
        self.std = self.model.std
        self.input_size = self.model.input_size[1] # assume every input is a square image
        self.input_range = self.model.input_range
        self.input_space = self.model.input_space
        self.resize_size = 354 # as in pretrainmodels repo
        
    def forward(self, x):
        # x = self.features(x)  
        # x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=0)
        # x = x.view(x.size(0), -1) 
        # x = x.dropout(training=self.training)
        # x = self.classifier(x) # 1x1000
        # return x
        return self.model.forward(x)
        
    def extract(self, x):
        # return self.features(x)
        return self.model.features(x)
    
def build(variant):
    net = Nasnet(variant).cuda()
    return net

architect='nasnet'