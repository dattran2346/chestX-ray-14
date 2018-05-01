import torchvision
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F

class DenseNet(nn.Module):
    
    def __init__(self, variant):
        super(DenseNet, self).__init__()
        assert variant in ['densenet121', 'densenet161', 'densenet201']
        model = pretrainedmodels.__dict__[variant](num_classes=1000, pretrained='imagenet')
        self.features = model.features
        num_ftrs = model.last_linear.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x) # 1x1024x7x7
        s = x.size()[3] # 7 if input image is 224x224, 16 if input image is 512x512
        x = F.relu(x, inplace=True) # 1x1024x7x7
        x = F.avg_pool2d(x, kernel_size=s, stride=1) # 1x1024x1x1
        x = x.view(x.size(0), -1) # 1x1024
        x = self.classifier(x) # 1x1000
        return x
        
    def extract(self, x):
        return self.features(x)

def build(variant):
    net = DenseNet(variant).cuda()
    return net

architect='densenet'
    
    