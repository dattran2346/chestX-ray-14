from utils import *
from constant import *
# from tensorboard import Tensorboard
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
import pdb
from tqdm import tqdm
from apex import amp


def main(args):
    print(args)
    amp.register_float_function(torch, 'sigmoid')
    amp_handle = amp.init()
    network = importlib.import_module(args.model_architect)
    # TODO: maybe get subdir from args, (in checkpoint mode)
    architect = network.architect
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    print('Model name %s' % subdir)
    model_dir = '%s/%s/%s/%s' % (args.models_base_dir, architect, args.model_variant, subdir)
    log_dir = '%s/%s/%s/%s' % (args.logs_base_dir, architect, args.model_variant, subdir)
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
    stat = {
        'train_loss': np.zeros((args.max_nrof_epochs,), np.float32),
        'train_auc': np.zeros((args.max_nrof_epochs,), np.float32),
        'val_loss': np.zeros((args.max_nrof_epochs,), np.float32),
        'val_auc': np.zeros((args.max_nrof_epochs,), np.float32),
        'lr': np.zeros((args.max_nrof_epochs), np.float32)
    }
    
    # TODO: load checkpoint model if exist
    # checkpoint = 
    
    # init training
    net = network.build(args.model_variant)
    optimizer = get_optimizer(net, args)
    
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    
    # TODO: Implement cyclic scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
    
    # Get data loader
    train_loader = train_dataloader(net, image_list_file=args.train_csv, percentage=0.3, batch_size=args.batch_size)
    # auc need sufficient large amount of either class to make sense, -> always load all here
    valid_loader = test_dataloader(net, image_list_file=args.val_csv, percentage=0.3, batch_size=args.batch_size)
    
    # start training
    best_dict = {
        'best_loss': float('inf')
    }
    
    log_lrs, losses = find_lr(net, train_loader, optimizer, criterion, args)
    print(log_lrs, losses)
    return
    
    # TODO: Add checkpoint
    for e in tqdm(range(args.max_nrof_epochs)):
        # train
        train_loss, train_auc = train(net, train_loader, optimizer, criterion, e, stat, args, amp_handle)
#         stat['train_auc'][e] = train_auc
#         stat['train_loss'][e] = train_loss
        
        # validate
        val_loss, val_auc = validate(net, valid_loader, criterion, e, stat, args)
#         stat['val_loss'][e] = val_loss
#         stat['val_auc'][e] = val_auc
        scheduler.step(val_loss)
           
        print(f'Epochs: [{e}/{args.max_nrof_epochs}]\tTrn_Loss: {train_loss} \tVal_Loss:{val_loss} \tAuc: {val_auc}')
        
        # print lr
#         for param_group in optimizer.param_groups:
#             stat['lr'][e] = param_group['lr']

        # save best model
        if val_loss < best_dict['best_loss']:
            best_dict = {
                'epoch': e+1,
                'state_dict': net.state_dict(),
                'best_loss': val_loss,
                'val_auc': val_auc,
                'optimizer': optimizer.state_dict()
            }
            torch.save(best_dict, model)
        
        # save stat
        with h5py.File(stat_file, 'w') as f:
            for key, value in stat.items():
                f.create_dataset(key, data=value)
    
    print('='*40)
    print('At epoch:', best_dict['epoch'])
    print('Min loss:', best_dict['best_loss'])
    print('Best AUC:', best_dict['val_auc'])
    print('='*40)
    print('Model name %s' % subdir)
    print('Args', args)

def find_lr(model, dataloader, optimizer, criterion, args, init_value=1e-8, final_value=10., beta=0.98):
    num = len(dataloader) - 1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for image, target in tqdm(dataloader):
        batch_num += 1
        image = Variable(torch.FloatTensor(image).cuda())
        target = Variable(torch.FloatTensor(target).cuda())
        optimizer.zero_grad()
        pred = model(image, pooling=args.pooling)
        loss = loss_func(criterion, pred, target, args)
        
        # compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta)*loss.item()
        smoothed_loss = avg_loss / (1-beta**batch_num)
        
        # stop if the loss is exploding
        if batch_num > 10 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        
        loss.backward()
        optimizer.step()
        
        lr *= mult
        
        optimizer.param_groups[0][lr] = lr
    return log_lrs, losses

def train(model, dataloader, optimizer, criterion, epoch, stat, args, amp_handle):
    model.train()
  
    losses = []
    targets = torch.FloatTensor().cuda()
    preds = torch.FloatTensor().cuda()
    
    for data in tqdm(dataloader):
        image, target = data
        image = Variable(torch.FloatTensor(image).cuda())
        target = Variable(torch.FloatTensor(target).cuda())
        
        optimizer.zero_grad()
        pred = model(image, pooling=args.pooling)
        
        # train with weighted loss
        loss = loss_func(criterion, pred, target, args)
        with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        optimizer.step()
        losses.append(loss.item())
        targets = torch.cat((targets, target.data), 0)
        preds = torch.cat((preds, pred.data), 0)
        
    aurocs = compute_aucs(targets.cpu(), preds.cpu())
    return np.mean(losses), np.mean(aurocs)
    

def validate(model, dataloader, criterion, epoch, stat, args):
    model.eval()
    losses = []
    targets = torch.FloatTensor().cuda()
    preds = torch.FloatTensor().cuda()
    
    for image, target in tqdm(dataloader):
        image = Variable(torch.FloatTensor(image).cuda())
        target = Variable(torch.FloatTensor(target).cuda())
        pred = model(image, pooling=args.pooling)
        loss = loss_func(criterion, pred, target, args)
        losses.append(loss.item())
        targets = torch.cat((targets, target.data), 0)
        preds = torch.cat((preds, pred.data), 0)
    aurocs = compute_aucs(targets.cpu(), preds.cpu())
    aurocs_mean = np.mean(aurocs)
   
    loss_mean = np.mean(losses)
    return np.mean(losses), aurocs_mean

def loss_func(criterion, pred, target, args):
    if args.loss == 'WBCE':
        # update weight
        y = target.cpu().data.numpy()
        y_hat = pred.cpu().data.numpy()
        P = np.count_nonzero(y == 1)
        N = np.count_nonzero(y == 0)
        beta_p = (P + N) / (P + 1) # may not contain disease 
        beta_n = (P + N) / N 
        w = np.empty(y.shape)
        w[y==0] = beta_n
        w[y==1] = beta_p
        w = torch.FloatTensor(w).cuda()
        criterion.weight = w
    return criterion(pred, target)

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
    parser.add_argument('--model_architect', type=str,
        help='Which architect to use', default='models.densenet')
    parser.add_argument('--model_variant', type=str,
        help='Variant of model, base on pretrainmodels', default='densenet121')
    
    # train process args
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=EPOCHS)
#     parser.add_argument('--epoch_size', type=int,
#         help='Number of batches per epoch.', default=BATCHES)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=BATCHSIZE)
    
    # train args
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate.', default=0.001)
    parser.add_argument('--pooling', type=str, choices=['MAX', 'AVG', 'LSE'],
        help='The pooling layer before final fc', default='AVG')
    parser.add_argument('--lse_r', type=float,
        help='Hyperparameter r of lse pooling', default='10')
    parser.add_argument('--loss', type=str, choices=['BCE', 'WBCE'],
        help='What loss function to use', default='BCE')
    
    # dataset args
    parser.add_argument('--train_csv', type=str,
        help='List of image to train in csv format', default=TRAIN_CSV)
    parser.add_argument('--val_csv', type=str,
        help='List of image to validate in csv format', default=VAL_CSV)
    
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    #TODO: Add argument parser?, or put in config file
    main(parse_arguments(sys.argv[1:]))