import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import SaveFeature
import pretrainedmodels
from torchvision.models import resnet50
from fastai.model import cut_model
from pathlib import Path
from torchvision.models.resnet import conv3x3, Bottleneck
from scipy import ndimage
import numpy as np
import torchvision.transforms as transforms
import cv2
from constant import IMAGENET_MEAN, IMAGENET_STD
from fastai.core import V


class UpBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, expansion=1):
        super().__init__()
        inplanes = inplanes * expansion
        planes = planes * expansion
        self.upconv = nn.ConvTranspose2d(inplanes, planes, 2, 2, 0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, u, x):
        up = self.relu(self.bn1(self.upconv(u)))
        out = torch.cat([x, up], dim=1) # cat along channel
        out = self.relu(self.bn2(self.conv1(out)))
        return out

class UpLayer(nn.Module):

    def __init__(self, block, inplanes, planes, blocks):
        super().__init__()
        self.up = UpBlock(inplanes, planes, block.expansion)
        layers = [block(planes*block.expansion, planes) for _ in range(1, blocks)]
        self.conv = nn.Sequential(*layers)

    def forward(self, u, x):
        x = self.up(u, x)
        x = self.conv(x)
        return x

class Unet(nn.Module):
    tfm = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    def __init__(self, trained=False, model_name=None):
        super().__init__()
        self.layers = [3, 4, 6]
        self.block = Bottleneck
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
        self.up_layer1 = UpLayer(self.block, 512, 256, self.layers[-1])
        self.up_layer2 = UpLayer(self.block, 256, 128, self.layers[-2])
        self.up_layer3 = UpLayer(self.block, 128, 64, self.layers[-3])

        self.map = conv3x3(64*self.block.expansion, 64) # 64e -> 64
        self.conv = conv3x3(128, 64)
        self.bn_conv = nn.BatchNorm2d(64)
        self.up_conv = nn.ConvTranspose2d(64, 1, 2, 2, 0)
        self.bn_up = nn.BatchNorm2d(1)

    def forward(self, x):
        x = F.relu(self.backbone(x))
        x = self.up_layer1(x, self.sfs[3].features)
        x = self.up_layer2(x, self.sfs[2].features)
        x = self.up_layer3(x, self.sfs[1].features)
        x = self.map(x)
        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([self.sfs[0].features, x], dim=1)
        x = F.relu(self.bn_conv(self.conv(x)))
        x = F.relu(self.bn_up(self.up_conv(x)))
        return x

    def close(self):
        for sf in self.sfs: sf.remove()

    def segment(self, image):
        """
        image: cropped CXR PIL Image (h, w, 3)
        """
        kernel = np.ones((10, 10))
        iw, ih = image.size

        image = V(self.tfm(image)[None])
        py = torch.sigmoid(self(image))
        py = (py[0].cpu() > 0.5).type(torch.FloatTensor) # 1, 256, 256

        mask = py[0].numpy()
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.resize(mask, (iw, ih))
        slice_y, slice_x = ndimage.find_objects(mask, 1)[0]
        h, w = slice_y.stop - slice_y.start, slice_x.stop - slice_x.start

        nw, nh = int(w/.875), int(h/.875)
        dw, dh = (nw-w)//2, (nh-h)//2
        t = max(slice_y.start-dh, 0)
        l = max(slice_x.start-dw, 0)
        b = min(slice_y.stop+dh, ih)
        r = min(slice_x.stop+dw, iw)
        return (t, l, b, r), mask
