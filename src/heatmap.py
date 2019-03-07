import torch
import numpy as np
import cv2
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

    def cam(self, pred_y):
        # (bs, 1024, 7, 7) * (bs, 1024) -> (bs, 7, 7) how??
        # heatmaps = []
        # for i, pred_y in enumerate(pred_ys):
        #     # permute: change axis order, reshape (view): mapping to new dimension
        #     heatmap = self.sf.features[i].permute(1, 2, 0) @ self.weight[pred_y]
        #     heatmaps.append(heatmap)
        # return torch.stack(heatmaps)
        heatmap = self.sf.features[0].permute(1, 2, 0) @ self.weight[pred_y]
        return heatmap

    # def default(self, pred_ys):
    #     return torch.max(torch.abs(self.sf.features), dim=1)[0]

    def generate(self, image):
        """
        input: PIL Image (h, w, c)
        output: heatmap np.array (h, w, 1)
        """
        prob = self.chexnet.predict(image)
        w, h = image.size
        return self.from_prob(prob, w, h)

    def from_prob(self, prob, w, h):
        """
        input: prob: np.array (14)
        output: heatmap np.array (h, w, 1)
        """
        pred_y = np.argmax(prob)
        heatmap = self.mapping(pred_y)

        # single image
        heatmap = to_np(heatmap)
        # heatmap = ((heatmap - heatmap.min())*(1./(heatmap.max() -
            # heatmap.min()) * 255.)).astype(np.uint8)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = cv2.resize(heatmap, (w, h))

        # multiple image
        # h, w = heatmap.shape
        # heatmap = heatmap.view(bs, h*w)
        # heatmap -= heatmap.min(1, keepdim=True)[0]
        # heatmap /= heatmap.max(1, keepdim=True)[0]
        # heatmap = heatmap.view(bs, h, w)

        return heatmap, np.take(CLASS_NAMES, to_np(pred_y))
