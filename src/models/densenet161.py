import torchvision
import torch.nn as nn

class DenseNet161(nn.Module):
    
    def __init__(self, out_size):
        super(DenseNet161, self).__init__()
        self.densenet161 = torchvision.models.densenet161(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet161.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.densenet161(x)
        
    def extract(self, x):
        if self.extractor is None:
            self.extractor = nn.Sequential(
                    *list(self.densenet161.features.children())[:-1]
                )
        return self.extractor.forward(x)

def build():
    net = DenseNet161(14).cuda()
    return net

architect='densenet161'