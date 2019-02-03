from torch import nn
import torch
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

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


class WeightedBCEWithLogitsLoss(nn.Module):

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

class SaveFeature:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()

# class FocalLoss(WeightedBCELoss):

#     def __init__(self, theta=2):
#         super().__init__()
#         self.theta = theta

#     def forward(self, input, target):
# #         pt = target*input + (1-target)*(1-input)
# #         target *= (1-pt)**self.theta
#         w = self.get_weight(input, target)
#         return F.binary_cross_entropy_with_logits(input, target, w)



# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()
