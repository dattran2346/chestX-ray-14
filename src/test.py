from constant import *
from utils import *
from statistic import TestStat
import torch
from torch.autograd import Variable
import numpy as np
import argparse
import sys
import importlib


def main(args):
    print(args)
    network = importlib.import_module(args.model_architect)
    architect = network.architect
    model_name = '%s/%s/%s/%s/model.path.tar' % (args.models_base_dir, architect, args.model_variant, args.model_name)
    net = network.build(args.model_variant)
    parallel_net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3]).cuda()
    
    checkpoint = torch.load(model_name)
    print('Best loss', checkpoint['best_loss'])
    print('AUC', checkpoint['aurocs_mean'])
    net.load_state_dict(checkpoint['state_dict'])
    test_loader = test_dataloader(image_list_file=args.test_csv, percentage=args.percentage, agumented=args.agumented)
    test(parallel_net, test_loader, args)


def test(model, dataloader, args):
    model.eval()
    targets = torch.FloatTensor()
    targets = targets.cuda()
    preds = torch.FloatTensor()
    preds = preds.cuda()
    stat = TestStat(args.model_name)
    
    for data, target in dataloader:
        target = target.cuda()
        if args.agumented:
            bs, cs, c, h, w = data.size()
            data = data.view(-1, c, h, w)
        data = Variable(data.cuda(), volatile=True)
        pred = model(data)
        if args.agumented:
            pred = pred.view(bs, cs, -1).mean(1)
        targets = torch.cat((targets, target), 0)
        preds = torch.cat((preds, pred.data), 0)
    aurocs = compute_aucs(targets, preds)
    aurocs_avg = np.array(aurocs).mean()
    
    print('The average AUROC is {0:.3f}'.format(aurocs_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], aurocs[i]))
    stat.stat(aurocs)

    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    # model args
    parser.add_argument('--model_name', type=str, 
        help='Model to test', default='20180422-075022')
    parser.add_argument('--models_base_dir', type=str,
        help='Where you put your model', default=MODEL_DIR)
    parser.add_argument('--model_architect', type=str,
        help='Which architect to use', default='models.densenet')
    parser.add_argument('--model_variant', type=str,
        help='Variant of model, base on pretrainmodels', default='densenet121')
    
    # data args
    parser.add_argument('--agumented',
        help='Agumented test data', action='store_true')
    parser.add_argument('--test_csv', type=str,
        help='List of image to test in csv format', default=CHEXNET_TEST_CSV)
    parser.add_argument('--percentage', type=float,
        help='Percentage of data to test', default=1.0) # default is test all
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    #TODO: Add argument parser?, or put in config file
    main(parse_arguments(sys.argv[1:]))

