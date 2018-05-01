import torchvision
import torch.nn as nn

class DenseNet201(nn.Module):
    
    def __init__(self, out_size):
        super(DenseNet201, self).__init__()
        self.densenet201 = torchvision.models.densenet201(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet201.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.densenet201(x)
        
    def extract(self, x):
        if self.extractor is None:
            self.extractor = nn.Sequential(
                    *list(self.densenet201.features.children())[:-1]
                )
        return self.extractor.forward(x)

def build():
    net = DenseNet201(14).cuda()
    return net

architect='densenet201'