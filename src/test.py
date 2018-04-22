from constant import *
from utils import *
from model import DenseNet121
from statistic import TestStat
import torch
from torch.autograd import Variable
import numpy as np

def main():
    # TODO: Pass name, architecture by arg
    name = '20180420-092017'
    model_name = '%s/%s/model.path.tar' % (DENSENET121_DIR, name)
    net = DenseNet121(N_CLASSES).cuda()
    parallel_net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    
    checkpoint = torch.load(model_name)
    print('Best loss', checkpoint['best_loss'])
    print('AUC', checkpoint['aurocs_mean'])
    parallel_net.load_state_dict(checkpoint['state_dict'])
    # TODO: Load test set, argment by arg
    test_loader = test_dataloader(image_list_file=CHEXNET_TEST_CSV, percentage=1, agumented=True)
    test(parallel_net, test_loader, agumented=True)


def test(model, dataloader, agumented=TEST_AGUMENTED):
    model.eval()
    targets = torch.FloatTensor()
    targets = targets.cuda()
    preds = torch.FloatTensor()
    preds = preds.cuda()
    name = '20180420-092017'
    stat = TestStat(name)
    
    for data, target in dataloader:
        target = target.cuda()
        if agumented:
            bs, cs, c, h, w = data.size()
            data = data.view(-1, c, h, w)
        data = Variable(data.cuda(), volatile=True)
        pred = model(data)
        if agumented:
            pred = pred.view(bs, cs, -1).mean(1)
        targets = torch.cat((targets, target), 0)
        preds = torch.cat((preds, pred.data), 0)
    aurocs = compute_aucs(targets, preds)
    aurocs_avg = np.array(aurocs).mean()
    
    print('The average AUROC is {0:.3f}'.format(aurocs_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], aurocs[i]))
    stat.stat(aurocs)
        
if __name__ == '__main__':
    #TODO: Add argument parser?, or put in config file
    main()