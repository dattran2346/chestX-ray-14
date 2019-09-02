from torch.optim.lr_scheduler import _LRScheduler
import math
from matplotlib import pyplot as plt
import torch
import numpy as np


# TODO: Implement ReduceOnPlateauScheduler for fastai
class LRFinder():

    def __init__(self, optimizer, start_lr, end_lr, batch_num, linear=False):
        self.optimizer = optimizer
        self.iteration = 0

        self.linear = linear
        self.start_lrs = np.array([start_lr] * len(optimizer.param_groups))
        ratio = end_lr / start_lr
        self.lr_mult = (ratio / batch_num) if linear else ratio**(1/batch_num)

        self.best = 1e9
        self.lrs = []
        self.metrics = []
        self.beta = 0.98
        self.avg_metric = 0.

        for group in optimizer.param_groups:
            group['lr'] = start_lr

    def step(self, metric):
        self.iteration += 1
        metric = self.cal_metric(metric)
        if math.isnan(metric) or metric > self.best*4:
            return False

        if metric < self.best and self.iteration > 10:
            self.best = metric
        self.metrics.append(metric.cpu().detach().numpy()) # to_np

        for param_group, lr in zip(self.optimizer.param_groups,
                self.get_lr()):
            param_group['lr'] = lr
        return True

    def cal_metric(self, metric):
        self.avg_metric = self.avg_metric*self.beta + metric*(1-self.beta)
        smoothed_metric = self.avg_metric / (1 - self.beta**self.iteration)
        return smoothed_metric

    def get_lr(self):
        mult = (self.lr_mult*self.iteration if self.linear else
                self.lr_mult**self.iteration)
        new_lrs = self.start_lrs*mult
        self.lrs.append(new_lrs[-1])
        return new_lrs

    def plot(self, skip_start, skip_end):
        plt.xlabel('Learning rate')
        plt.ylabel('Metric')
        plt.plot(self.lrs[skip_start:-(skip_end+1)],
                self.metrics[skip_start:-(skip_end+1)])
        plt.xscale('log')
