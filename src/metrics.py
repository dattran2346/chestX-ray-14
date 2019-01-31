from sklearn.metrics import roc_auc_score
import numpy as np
from fastai.core import to_np, T


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
    preds = (preds > 0).float()
    return 2. * (preds * targets).sum() / (preds + targets).sum()

def dice_np(preds, targets):
    preds = preds > 0
    return np.sum(preds * targets) / np.sum(preds + targets)
