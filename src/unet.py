import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import SaveFeature
import pretrainedmodels
from torchvision.models import resnet34, resnet50, resnet101, resnet152
from fastai.model import cut_model
from pathlib import Path

class UnetBlock(nn.Module):

    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2 # n_out is concat of up_out and x_out
        self.x_conv = nn.Conv2d(x_in, x_out, 1) # 1x1 conv
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1) # concat along #filter
        return self.bn(F.relu(cat_p))


class Unet(nn.Module):

    def __init__(self, trained=False, model_name=None):
        super().__init__()

        if trained:
            assert model_name != None
            self.load_model(model_name)
        else:
            self.load_pretrained()

    def load_model(self, model_name):
        resnet = resnet50(False)
        self.backbone = nn.Sequential(*cut_model(resnet, 8))
        self.init_head()

        path = Path('/home/dattran/data/xray-thesis/LungSegmentation/models')
        state_dict = torch.load(path/model_name/'best.h5')
        self.load_state_dict(state_dict)

    def load_pretrained(self, torch=False):
        if torch:
            resnet = resnet50(True)
        else:
            resnet = pretrainedmodels.__dict__['resnet50']()
        self.backbone = nn.Sequential(*cut_model(resnet, 8))
        self.init_head()

    def init_head(self):
        self.sfs = [SaveFeature(self.backbone[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.backbone(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
#         return x[:,0]
        return x

    def close(self):
        for sf in self.sfs: sf.remove()
