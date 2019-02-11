from sklearn.metrics import roc_auc_score
import numpy as np
from fastai.core import to_np, T
import torch


def aucs(preds, targets):
    targets = to_np(targets)
    preds = to_np(preds)

    aurocs = []
    n = preds.shape[1]
    print(n)
    # preds = np.nan_to_num(preds)  # some numpy was nan or inf
    for i in range(n):
        aurocs.append(roc_auc_score(targets[:, i], preds[:, i]))
    return T(aurocs)

def aucs_np(preds, targets):
    aurocs = []
    # preds = np.nan_to_num(preds)  # some numpy was nan or inf
    n = preds.shape[1]
    for i in range(n):
        aurocs.append(roc_auc_score(targets[:, i], preds[:, i]))
    return aurocs


def auc(preds, targets):
    return aucs(preds, targets).mean()

def auc_np(preds, targets):
    return np.mean(aucs_np(preds, targets))

def dice(preds, targets):
    """
    preds: tensor (output of nn after sigmoid)
    """
    assert targets.max().item() == 1
    assert targets.min().item() == 0

    preds = (preds > 0.5).float()
    return 2. * (preds * targets).sum() / (preds + targets).sum()

def dice_np(preds, targets):
    assert targets.max() == 1
    assert targets.min() == 0

    return np.sum(preds * targets) / np.sum(preds + targets)

def iou(preds, targets):
    assert targets.max().item() == 1
    assert targets.min().item() == 0

    x = (preds > 0.5).type(torch.ByteTensor)
    y = targets.type(torch.ByteTensor)

    intersect = (x & y).sum().item()
    union = (x | y).sum().item()
    return (intersect+1)*1.0 / (union+1)

def iou_np(preds, targets):
    assert targets.max() == 1
    assert targets.min() == 0

    x = preds > 0.5
    y = targets

    intersect = (x & y).sum()
    union = (x | y).sum()
    return (intersect+1) * 1.0 / (union+1)



