from constant import *
import pandas as pd
import numpy as np

class TestStat(object):
    
    def __init__(self, name, file=TEST_STAT):
        self.df = pd.read_csv(file)
        self.file = file
        self.name = name
    
    def stat(self, aurocs):
        mean = np.array(aurocs).mean()
        cols = ['Model'] + CLASS_NAMES + ['Mean']
        values = [self.name] + aurocs + [mean]
        row = pd.Series(values, index=cols)
        self.df = self.df.append(row, ignore_index=True)
        self.df.to_csv(self.file, index=False)
        
class TrainStat(object):
    
    def __init__(self, file=TRAIN_STAT):
        self.df = pd.read_csv(file)
        self.file = file
        self.row = {}
    
    def with_model_name(self, name):
        self.row['model_name'] = name
        return self
    
    def with_architecture(self, architecture):
        self.row['architecture'] = architecture
        return self
    
    def with_gpu(self, gpu):
        self.row['gpu'] = gpu
        return self
    
    def with_gpu_allocate(self, gpu_allocate):
        self.row['gpu_allocate'] = gpu_allocate
        return self
    
    def with_epochs(self, epochs):
        self.row['epochs'] = epochs
        return self
    
    def with_batches(self, batches):
        self.row['batches'] = batches
        return self
    
    def with_train_batch(self, train_batch):
        self.row['train_batch'] = train_batch
        return self
    
    def with_valid_batch(self, valid_batch):
        self.row['valid_batch'] = valid_batch
        return self
    
    def with_train_worker(self, train_worker):
        self.row['train_worker'] = train_worker
        return self
    
    def with_valid_worker(self, valid_worker):
        self.row['valid_worker'] = valid_worker
        return self
    
    def with_train_percentage(self, train_percentage):
        self.row['train_percentage'] = train_percentage
        return self
    
    def with_valid_percentage(self, valid_percentage):
        self.row['valid_percentage'] = valid_percentage
        return self

    def with_loss_function(self, criterion):
        name = type(criterion).__name__.split('.')[-1]
        self.row['loss_function'] = name
        return self
    
    def with_optimizer(self, optimizer):
        '''
        Log the optimizer param before training
        '''
        name = type(optimizer).__name__.split('.')[-1]
        # param_groups key is not change before, during and after training
        state = optimizer.state_dict()['param_groups'][-1]
        state.pop('params')
        op = '%s(%s)' % (name, state)
        self.row['optimizer'] = op
        return self
    
    def with_scheduler(self, scheduler):
        name = type(scheduler).__name__.split('.')[-1]
        self.row['scheduler'] = name
        return self
    
    def with_train_time(self, train_time):
        self.row['train_time'] = train_time
        return self
    
    def with_test_time(self, test_time):
        self.row['test_time'] = test_time
        return self
    
    def with_test_auc(self, test_auc):
        self.row['test_auc'] = test_auc
        return self
    
    def with_note(self, note):
        self.row['note'] = note
        return self
    
    def commit(self):
        self.df = self.df.append(self.row, ignore_index=True)
        self.row = {}
        print(self.df)
        self.df.to_csv(self.file, index=False)
        

# stat = TestStat(name='test-model')
# aurocs = [0.8311,0.922,0.8891,0.7146,0.8627,0.7883,0.782,0.8844,0.8148,0.8992,0.9343,0.8385,0.7914,0.9206]
# stat.stat(aurocs)

# stat = TrainStat()
# (stat.with_model_name('test-model')
#      .with_architecture('densenet121')
#      .commit())
        