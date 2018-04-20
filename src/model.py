import torchvision
import torch.nn as nn

class DenseNet121(nn.Module):
    
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        #print('\tForward')
        x = self.densenet121(x)
        return x
        
    def extract(self, x):
        if self.extractor is None:
            self.extractor = nn.Sequential(
                    *list(self.densenet121.features.children())[:-1]
                )
        return self.extractor.forward(x)
    
    