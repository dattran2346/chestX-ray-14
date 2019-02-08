import torch.nn as nn
import pretrainedmodels
from torchvision.models import densenet121
from models.densenet import DenseNet # old stuff
from layers import Flatten
import torch
from pathlib import Path
from fastai.torch_imports import children

class ChexNet(nn.Module):

    def __init__(self, trained=False, model_name='20180525-222635'):
        super().__init__()
        # chexnet.parameters() is freezed except head
        if trained:
            # self.load_prethesis(model_name)
            self.load_model(model_name)
        else:
            self.load_pretrained()

    # def load_prethesis(self, model_name):
    #     # load pre-thesis model
    #     densenet = DenseNet('densenet121')
    #     path = Path('/mnt/data/xray-thesis/models/densenet/densenet121')
    #     checkpoint = torch.load(path/model_name/'model.path.tar')
    #     densenet.load_state_dict(checkpoint['state_dict'])
    #     self.backbone = densenet.features
    #     self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
    #                               Flatten(),
    #                               children(densenet.classifier)[0])

    def load_model(self, model_name):
        self.backbone = densenet121(False).features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 Flatten(),
                                 nn.Linear(1024, 14))
        path = Path('/home/dattran/data/xray-thesis/chestX-ray14/models')
        state_dict = torch.load(path/model_name/'best.h5')
        self.load_state_dict(state_dict)

    def load_pretrained(self, torch=False):
        if torch:
            # torch vision, train the same -> ~0.75 AUC on test
            self.backbone = densenet121(True).features
        else:
            # pretrainmodel, train -> 0.85 AUC on test
            self.backbone = pretrainedmodels.__dict__['densenet121']().features

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  Flatten(),
                                  nn.Linear(1024, 14))

    def forward(self, x):
        return self.head(self.backbone(x))
