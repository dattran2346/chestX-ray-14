import torch
import numpy as np
from chexnet import ChexNet
from layers import SaveFeature
from constant import CLASS_NAMES
from fastai.core import to_np


class HeatmapGenerator:

    # def __init__(self, model_name='20180429-130928', mode=None):
    def __init__(self, chexnet, mode=None):
        # init chexnet and register forward hook
        # self.chexnet = ChexNet(model_name='20180429-130928')
        # self.sf = SaveFeature(self.chexnet.backbone)
        self.chexnet = chexnet
        self.sf = SaveFeature(chexnet.backbone)
        self.weight = list(list(self.chexnet.head.children())[-1].parameters())[0]
        self.mapping = self.cam if mode=='cam' else self.default

    def cam(self, pred_ys):
        # (bs, 1024, 7, 7) * (bs, 1024) -> (bs, 7, 7) how??
        heatmaps = []
        for i, pred_y in enumerate(pred_ys):
            # permute: change axis order, reshape (view): mapping to new dimension
            heatmap = self.sf.features[i].permute(1, 2, 0) @ self.weight[pred_y]
            heatmaps.append(heatmap)
        return torch.stack(heatmaps)

    def default(self, pred_ys):
        return torch.max(torch.abs(self.sf.features), dim=1)[0]

    def generate(self, images):
        """
        input: tensor (bs, c, h, w)
        output: tensor (bs, h/32, w/32)
        """
        py = torch.sigmoid(self.chexnet(images))
        pred_ys = torch.argmax(py, dim=1)
        heatmaps = self.mapping(pred_ys)
        bs, h, w = heatmaps.shape

        # scale to [0, 1]
        heatmaps = heatmaps.view(bs, h*w)
        heatmaps -= heatmaps.min(1, keepdim=True)[0]
        heatmaps /= heatmaps.max(1, keepdim=True)[0]
        heatmaps = heatmaps.view(bs, h, w)
        return heatmaps, np.take(CLASS_NAMES, to_np(pred_ys))

