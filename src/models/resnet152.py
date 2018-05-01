import torchvision
import torch.nn as nn

class Resnet152(nn.Module):
    
    def __init__(self, out_size):
        super(Resnet152, self).__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained=True)
        num_ftrs = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet152(x)
    
def build():
    net = Resnet152(14).cuda()
    return net

architect='resnet152'