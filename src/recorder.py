from collections import defaultdict
import pickle
from time import time
import numpy as np
from fastai.sgdr import LossRecorder


class TrainingRecoder(LossRecorder):

    # This training recoder with save state between training phase
    def __init__(self, save_path='', record_mom=False):
        self.layer_opt = None
        self.save_path, self.record_mom = save_path, record_mom

        # record every iteration
        self.losses, self.lrs, self.iterations = [], [], []
        self.epoch = 0

        # record every batch
        self.epochs = []
        self.val_losses, self.trn_losses, self.rec_metrics = [], [], [] # save validate
        self.iteration = 0

        self.epoch_losses = []

        if self.record_mom:
            self.momentums = []

    def new_phase(self, layer_opt):
        # call this before every learn.fit_gen to set new layer_opt
        self.layer_opt = layer_opt

    def on_train_begin(self):
        pass

    def on_epoch_begin(self):
        self.epoch += 1


    def on_epoch_end(self, metrics):
        # record trn_loss and reset cache
        self.trn_losses.append(np.mean(self.epoch_losses))
        self.epoch_losses = []

        self.save_metrics(metrics) # [val_loss + metrics]

    def on_batch_end(self, loss):
        self.epoch_losses.append(loss)
        super().on_batch_end(loss)

    def save(self):
        stat = {
            'trn_losses': self.trn_losses,
            'val_losses': self.val_losses,
            'metrics': np.stack(self.rec_metrics)
        }
        with open(self.save_path/'log.pickle', 'wb') as f:
            pickle.dump(stat, f)
