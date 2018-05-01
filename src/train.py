from utils import *
from constant import *
from tensorboard import Tensorboard
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from datetime import datetime
import os
import argparse
import sys
import importlib
import h5py


def main(args):
    print(args)
    network = importlib.import_module(args.model_def)
    architect = network.architect
    # TODO: maybe get subdir from args, (in checkpoint mode)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    print('Model name %s' % subdir)
    model_dir = '%s/%s/%s' % (args.models_base_dir, architect, subdir)
    log_dir = '%s/%s/%s' % (args.logs_base_dir, architect, subdir)
    print('Model dir', model_dir)
    print('Log dir', log_dir)
    
    # check and create dir if not existed
    dirs = [model_dir, log_dir]
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)
    model = '%s/model.path.tar' % model_dir
    stat_file = '%s/stat.h5' % log_dir
    
    # init logger
    board = Tensorboard(log_dir)
    stat = {
        'train_loss': np.zeros((args.max_nrof_epochs,), np.float32),
        'train_auc': np.zeros((args.max_nrof_epochs,), np.float32),
        'val_loss': np.zeros((args.max_nrof_epochs,), np.float32),
        'val_auc': np.zeros((args.max_nrof_epochs,), np.float32)
    }
    
    # TODO: load checkpoint model if exist
    # checkpoint = 
    
    # init training
    net = network.build()
    parallel_net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    optimizer = get_optimizer(parallel_net, args)
    
    # TODO: Try different loss function
    criterion = nn.BCELoss()
    
    # TODO: Try different scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
    
    # Get data loader
    train_loader = train_dataloader(image_list_file=args.train_csv, percentage=1)
    # auc need sufficient large amount of either class to make sense, -> always load all here
    valid_loader = test_dataloader(image_list_file=args.val_csv, percentage=1, agumented=args.agumented)
    
    # start training
    batches = min(args.epoch_size, len(train_loader))
    loss_min = float('inf')
    # TODO: Add checkpoint
    for e in range(args.max_nrof_epochs):
        # train
        train(parallel_net, train_loader, optimizer, criterion, e, batches, stat)
        
        # validate
        loss_val, aurocs_mean = validate(parallel_net, valid_loader, criterion, e, stat)
        scheduler.step(loss_val)

        # save best model
        if loss_val < loss_min:
            loss_min = loss_val
            torch.save({
                'epoch': e+1,
                'state_dict': parallel_net.state_dict(),
                'best_loss': loss_min,
                'aurocs_mean': aurocs_mean,
                'optimizer': optimizer.state_dict()
            }, model)
        
        # save stat
        with h5py.File(stat_file, 'w') as f:
            for key, value in stat.items():
                f.create_dataset(key, data=value)
                
    print('Model name %s' % subdir)
    print('Args', args)
    

def train(model, dataloader, optimizer, criterion, epoch, batches, stat):
    model.train()
    iterator = iter(dataloader)
    stime = time.time()
    
    losses = []
    targets = torch.FloatTensor().cuda()
    preds = torch.FloatTensor().cuda()
    
    for i in range(batches):
        data, target = iterator.next()
        data = Variable(torch.FloatTensor(data).cuda())
        target = Variable(torch.FloatTensor(target).cuda())
        
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        duration = time.time() - stime
        print('Epochs: [%d][%d/%d]\tTime: %.3f \tLoss: %2.3f' % (epoch, i+1, batches, duration, loss))
        stime += duration
        
        losses.append(loss.data[0])
        targets = torch.cat((targets, target.data), 0)
        preds = torch.cat((preds, pred.data), 0)
        
        # board.scalar_summary('train_loss', loss.data, epoch * batches + i + 1)
    aurocs = compute_aucs(targets, preds)
    stat['train_auc'][epoch] = np.mean(aurocs)
    stat['train_loss'][epoch] = np.mean(losses)
    
        

def validate(model, dataloader, criterion, epoch, stat):
    model.eval()
    losses = []
    targets = torch.FloatTensor().cuda()
    preds = torch.FloatTensor().cuda()
    
    for data, target in dataloader:
        data = Variable(torch.FloatTensor(data).cuda(), volatile=True)
        target = Variable(torch.FloatTensor(target).cuda(), volatile=True)
        pred = model(data)
        loss = criterion(pred, target)
        losses.append(loss.data[0])
        targets = torch.cat((targets, target.data), 0)
        preds = torch.cat((preds, pred.data), 0)
    aurocs = compute_aucs(targets, preds)
    aurocs_mean = np.mean(aurocs)
    print('The average AUROC is %.3f' % aurocs_mean)
    
    loss_mean = np.mean(losses)
    stat['val_loss'][epoch] = loss_mean
    stat['val_auc'][epoch] = aurocs_mean
    return np.mean(losses), aurocs_mean


def get_optimizer(net, args):
    optim_name = args.optimizer
    lr = args.learning_rate
    if optim_name == 'ADAGRAD':
        # Adaptive Subgradient Methods for Online Learning and Stochastic Optimizatio
        optimizer = optim.Adagrad(net.parameters(), lr=lr, lr_decay=0, weight_decay=0)
    elif optim_name == 'ADADELTA':
        # ADADELTA: An Adaptive Learning Rate Method
        optimizer = optim.Adadelta(net.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
    elif optim_name == 'ADAM':
        # Adam: A Method for Stochastic Optimization
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optim_name == 'RMSPROP':
        '''
        Proposed by G. Hinton in his
        `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>
        '''
        optimizer = optim.RMSprop(net.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)
    else:
        raise ValueError('Invalid optimization algorithm')
    return optimizer
        
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    # directory args
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default=LOG_DIR)
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default=MODEL_DIR)
    parser.add_argument('--model_def', type=str,
        help='Directory where to write trained models and checkpoints.', default='models.densenet121')
    
    # train process args
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=EPOCHS)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=BATCHES)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=BATCHSIZE)
    
    # train args
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate.', default=0.001)
    
    # dataset args
    parser.add_argument('--train_csv', type=str,
        help='List of image to train in csv format', default=CHEXNET_TRAIN_CSV)
    parser.add_argument('--val_csv', type=str,
        help='List of image to validate in csv format', default=CHEXNET_VAL_CSV)
    parser.add_argument('--agumented',
        help='Agumented validate data', action='store_true')
    
    # 
    
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    #TODO: Add argument parser?, or put in config file
    main(parse_arguments(sys.argv[1:]))