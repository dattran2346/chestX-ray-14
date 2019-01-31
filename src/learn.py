from datetime import datetime
import torch.optim as optim
from torch.autograd import Variable
import torch
from apex import amp
import numpy as np
from metrics import aucs
import torch.optim.lr_scheduler as scheduler
from scheduler import LRFinder
from tqdm import tqdm, tnrange


class Learner(object):

    def __init__(self, model, dataloader, criterion, recorder, path,
            model_name,  *args):
        amp.register_float_function(torch, 'sigmoid')
        self.amp_handle = amp.init()

        self.model = model
        self.recorder = recorder
        self.dataloader = dataloader
        self.save_path = path/model_name
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.opt_fn = optim.Adam
        self.criterion = criterion
        self.sched = None

    def find_lr(self, start_lr=1e-5, end_lr=100, linear=False):
        self.optimizer = self.opt_fn(self.model.parameters(), lr=start_lr)
        self.sched = LRFinder(self.optimizer, start_lr, end_lr,
                len(self.dataloader.train), linear)

        self.save('tmp')
        for image, target in tqdm(self.dataloader.train):
            image = Variable(torch.FloatTensor(image).cuda())
            target = Variable(torch.FloatTensor(target).cuda())
            pred = self.model(image)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()
            do = self.sched.step(loss)

            if not do:
                break
        self.load('tmp')

    def fit(self, lrs, epochs, metrics=[]):
        self.optimizer = self.opt_fn(self.model.parameters(), lrs)
        self.sched = scheduler.CosineAnnealingLR(self.optimizer, 1000)

        names = ['epoch', 'trn_loss', 'val_loss']
        for f in metrics:
            names += [f'trn_{f.__name__}', f'val_{f.__name__}']
        layouts = '{!s:10} ' * len(names)
        pre_vals = []

        self.recorder.on_start()
        for e in tnrange(epochs, desc='Epoch'):
            trn_values = self.train(metrics)  # [loss, metrics...]
            val_values = self.validate(metrics)  # [loss, metrics...]

            values = [None] * (len(trn_values) + len(val_values))
            values[::2] = trn_values
            values[1::2] = val_values

            self.recorder.on_epoch_end(**dict(zip(names[1:], values)))

            if e == 0:
                print(layouts.format(*names))
                self.print_stat(e, values)
            else:
                self.print_stat(e, values, pre_vals)

            pre_vals = values
        self.recorder.on_end()

    def train(self, metrics):
        self.model.train()

        targets = None
        preds = None
        losses = []
        t = tqdm(self.dataloader.train,
                desc='Train', leave=False, miniters=0)
        for image, target in t:
            image = Variable(torch.FloatTensor(image).cuda())
            target = Variable(torch.FloatTensor(target).cuda())

            self.optimizer.zero_grad()
            pred = self.model(image)
            loss = self.criterion(pred, target)
            losses.append(loss.cpu().data)

            t.set_postfix(loss=loss.item(), refresh=False)

            with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()
            self.sched.step()  # step per batch, not per epoch

            if targets is None:
                targets = target.cpu().data
                preds = pred.cpu().data
            else:
                targets = np.concatenate((targets, target.cpu().data), axis=0)
                preds = np.concatenate((preds, pred.cpu().data), 0)

        t.close()

        results = [f(targets, preds) for f in metrics]
        loss = np.average(losses)
        return [loss] + results

    def validate(self, metrics):
        self.model.eval()

        targets = None
        preds = None
        losses = []
        t = tqdm(self.dataloader.val,
                desc='Validate', leave=False, miniters=0)
        for image, target in t:
            image = Variable(torch.FloatTensor(image).cuda())
            target = Variable(torch.FloatTensor(target).cuda())

            pred = self.model(image)
            loss = self.criterion(pred, target)
            losses.append(loss.cpu().data)

            if targets is None:
                targets = target.cpu().data
                preds = pred.cpu().data
            else:
                targets = np.concatenate((targets, target.cpu().data), axis=0)
                preds = np.concatenate((preds, pred.cpu().data), 0)

        t.close()

        results = [f(targets, preds) for f in metrics]
        loss = np.average(losses)
        return [loss] + results

    def test(self, metrics):
        self.model.eval()

        targets = np.array()
        preds = np.array()
        # losses = []

        for image, target in tqdm(self.dataloader.test):
            image = torch.cuda.HaftTensor(image)
            target = torch.cuda.HaftTensor(target)
            pred = self.model(image)

            targets = np.concatenate((targets, target.cpu().data), axis=0)
            preds = np.concatenate((preds, pred.data.cpu().data), 0)

        results = [f(targets, preds) for f in metrics]
        return results

    def print_stat(self, epoch, values, pre_vals=[]):
        sym = ""
        if epoch == 0: sym = ""
        elif values[0] > pre_vals[0] and values[1] > pre_vals[1]: sym = " △ △"
        elif values[0] > pre_vals[0] and values[1] < pre_vals[1]: sym = " △ ▼"
        elif values[0] < pre_vals[0] and values[1] < pre_vals[1]: sym = " ▼ ▼"
        elif values[0] < pre_vals[0] and values[1] > pre_vals[1]: sym = " ▼ △"

        layout = "{!s:^10}" + " {!s:10}" * len(values)
        values = [epoch] + list(np.round(values, 6))
        print(layout.format(*values) + sym)

    def predict(self, images):
        print("TODO: Predict API")

    def save(self, name):
        torch.save(self.model.state_dict(), f'{self.save_path}/{name}.pth')

    def load(self, name):
        state_dict = torch.load(f'{self.save_path}/{name}.pth')
        self.model.load_state_dict(state_dict)

