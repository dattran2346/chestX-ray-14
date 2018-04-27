import torchvision
import torch.nn as nn

class Resnet50(nn.Module):
    
    def __init__(self, out_size):
        super(Resnet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet50(x)
    
def build():
    net = Resnet50(14).cuda()
    return net

architect='resnet50'