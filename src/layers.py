from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class LSEPool2d(nn.Module):
    
    def __init__(self, r=3):
        super().__init__()
        self.r =r
    
    def forward(self, x):
        s = x.size()[3]  # x: bs*2048*7*7
        r = self.r
        x_max = F.adaptive_max_pool2d(x, 1) # x_max: bs*2048*1*1
        p = ((1/r) * torch.log((1 / (s*s)) * torch.exp(r*(x - x_max)).sum(3).sum(2)))
        x_max = x_max.view(x.size(0), -1) # bs*2048
        return x_max+p


class WeightedBCELoss(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        w = self.get_weight(input, target)
        return F.binary_cross_entropy_with_logits(input, target, w, reduction='mean')
    
    def get_weight(self, input, target):
        y = target.cpu().data.numpy()
        y_hat = input.cpu().data.numpy()
        P = np.count_nonzero(y == 1)
        N = np.count_nonzero(y == 0)
        beta_p = (P + N) / (P + 1) # may not contain disease 
        beta_n = (P + N) / N 
        w = np.empty(y.shape)
        w[y==0] = beta_n
        w[y==1] = beta_p
        w = torch.FloatTensor(w).cuda()
        return w