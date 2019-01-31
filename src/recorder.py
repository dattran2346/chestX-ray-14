from collections import defaultdict
import pickle
import h5py
from time import time
import matplotlib.pyplot as plt
import seaborn as sns


class LearningRecorder():

    def __init__(self, path, model_name):
        self.save_path = path/model_name
        self.save_path.mkdir(parents=True, exist_ok=True)

    def init_log(self):
        self.metric_dict = defaultdict(lambda: [])

    def on_start(self):
        self.init_log()
        self.time_start = time()

    def on_end(self):
        self.time_elapse = time() - self.time_start

    def on_epoch_end(self, **metrics):
        for k, v in metrics.items():
            self.metric_dict[k].append(v)

    def save(self, name):
        log = {
            'time_elapse': self.time_elapse,
            'epoch_log': dict(self.metric_dict)  # convert to normal dict (lambda is not pickable)
        }

        with open(f'{self.save_path}/{name}.pickle', 'wb') as f:
            pickle.dump(log, f)

    def load(self, name):
        with open(f'{self.save_path}/{name}.pickle', 'rb') as f:
            log = pickle.load(f)
            self.metric_dict = log['epoch_log']

    def plot(self, metrics):
        colors = sns.color_palette(n_colors=len(metrics))
        for metric, c in zip(metrics, colors):
            trn = self.metric_dict[f'trn_{metric}']
            val = self.metric_dict[f'val_{metric}']
            plt.figure()
            plt.plot(trn, c=c, label='Train')
            plt.plot(val, c=c, dashes=[6, 2], label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()

