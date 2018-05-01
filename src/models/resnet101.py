import torchvision
import torch.nn as nn

class Resnet101(nn.Module):
    
    def __init__(self, out_size):
        super(Resnet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet101(x)
    
def build():
    net = Resnet101(14).cuda()
    return net

architect='resnet101'