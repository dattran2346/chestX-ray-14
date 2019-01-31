
# coding: utf-8

# In[33]:




# In[34]:


from fastai.conv_learner import *
from fastai.dataset import *
from fastai.transforms import *
from fastai.models.resnet import vgg_resnet50
from sklearn.metrics import roc_auc_score
# import pixiedust
from matplotlib.patches import Rectangle
from datetime import datetime
import pandas as pd
from constant import CLASS_NAMES
from layers import *
import torch
from transform import chest_xray_transfrom
from metrics import *
import pretrainedmodels


# In[42]:


PATH = Path('/home/dattran/data/xray-thesis/chestX-ray14')
IMAGE_DN = 'images'
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
TEST_CSV = 'test.csv'

arch = densenet121
percentage = 1
WORKERS = 6

N_CLASSES = 14
models_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

model_path = PATH/'models'/models_name
model_path.mkdir(parents=True, exist_ok=True)
print('Model name', models_name)


# ## Load dataset

# In[43]:


from PIL import Image
import pretrainedmodels
from torch.utils.data import DataLoader
from dataset import ChestXray14Dataset
from model_data import ModelData

def get_chestxray_from_csv():
    result = []
    for f in [PATH/TRAIN_CSV, PATH/VAL_CSV, PATH/TEST_CSV]:
        df = pd.read_csv(f, sep=' ', header=None)
        images = df.iloc[:, 0].values
        labels = df.iloc[:, 1:].values
        result.append((images, labels))
    return result

def get_md(sz, bs, percentage):
    model = pretrainedmodels.__dict__['densenet121']()
    data = get_chestxray_from_csv()
    tfms = chest_xray_transfrom(model, sz, 0.875)
    datasets = ImageData.get_ds(ChestXray14Dataset, trn=data[0], val=data[1], test=data[2], tfms=tfms, path=PATH/IMAGE_DN, size=sz, percentage=percentage)
    md = ModelData(PATH/IMAGE_DN, datasets, bs, WORKERS, CLASS_NAMES)
    return md

md = get_md(224, 128, percentage)


# ## Densenet121 arch with binary classification

# In[44]:


from models.densenet import DenseNet

class ChexNet(nn.Module):

    def __init__(self):
        super().__init__()
        # chexnet.parameters() is freezed except head
        self.load_pretrained()

    def load_prethesis(self, model_name='20180525-222635'):
        # load pre-thesis model
        densenet = DenseNet('densenet121')
        path = Path('/mnt/data/xray-thesis/models/densenet/densenet121')
        checkpoint = torch.load(path/model_name/'model.path.tar')
        densenet.load_state_dict(checkpoint['state_dict'])
        self.backbone = densenet.features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  Flatten(),
                                  children(densenet.classifier)[0])

    def load_pretrained(self, torch=False):
        if torch:
            # torch vision, train the same -> ~0.75 AUC on test
            self.backbone = densenet121(True).features
        else:
            # pretrainmodel, train -> 0.85 AUC on test
            self.backbone = pretrainedmodels.__dict__['densenet121']().features

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  Flatten(),
                                  nn.Linear(1024, 14))

    def forward(self, x):
        return self.head(self.backbone(x))

class CheXNetModel():

    def __init__(self, chexnet):
        self.model = chexnet

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.backbone), [8]))
        lgs += [children(self.model.head)]
        return lgs

chexnet = to_gpu(ChexNet())
chexnet_model = CheXNetModel(chexnet)
learn = ConvLearner(md, chexnet_model, models_name=model_path) # train in mixed percision
# learn.crit = WeightedBCEWithLogitsLoss()
learn.crit = nn.BCEWithLogitsLoss()


# In[ ]:


### Warning ###


# # Find lr for last layer groups

# In[45]:


# learn.lr_find()


# In[46]:


# learn.sched.plot(0, 0)


# # Lr find for all layer group

# In[47]:


# md = get_md(224, 16, percentage)
# learn.set_data(md)
# learn.unfreeze()
# learn.lr_find()


# In[48]:


# learn.sched.plot(0, 0)


# ## Develop

# In[19]:


# the model we use not include sigmoid as last layer
def acc_s(preds, targs):
    return accuracy_multi(torch.sigmoid(preds), targs, 0.5)

def auc_s(preds, targs):
    return auc(torch.sigmoid(preds), targs)


# In[20]:


import numpy as np
class TrainingRecoder(LossRecorder):
    # This training recoder with save state between training phase
    def __init__(self, save_path='', record_mom=False):
        self.layer_opt = None
        self.save_path, self.record_mom = save_path, record_mom

        # record every iteration
        self.losses,self.lrs,self.iterations = [], [], []
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
#         self.iteration += 1
#         self.lrs.append(self.layer_opt.lr)
#         if isinstance(loss, list):
#             self.losses.append(loss[0])
#             self.save_metrics(loss[1:])
#         else: self.losses.append(loss)
#         if self.record_mom: self.momentums.append(self.layer_opt.mom)

    def save(self):
        import pickle
        import numpy as np
        stat = {
            'trn_losses': self.trn_losses,
            'val_losses': self.val_losses,
            'metrics': np.stack(self.rec_metrics)
        }
        with open(self.save_path/'log.pickle', 'wb') as f:
            pickle.dump(stat, f)



# In[9]:


# lr = 1e-4
# md = get_md(224, 128, percentage)
# learn.set_data(md)


# In[10]:


# layer_opt = learn.get_layer_opt(lr, None)
# loss_recorder = LossRecorder(layer_opt)
# learn.sched = None
# # learn.fit_gen(learn.model, learn.data, layer_opt, 2, \
# #                   metrics=[auc_s, acc_s], cycle_len=1, use_clr=(20, 10), \
# #                   callbacks=[loss_recorder], best_save_name='best')
# learn.fit_gen(learn.model, learn.data, layer_opt, 2, use_clr=(20, 10), cycle_len=1)


# In[ ]:


# # Test pytorch train to check why fastai loss is too big??
# model = learn.model
# iterator = iter(md.trn_dl)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
# # criterion = nn.BCEWithLogitsLoss()
# criterion = WeightedBCEWithLogitsLoss()

# for e in range(2):
#     losses = []
#     for b in range(100):
#         images, targets = iterator.next()
#         images = Variable(torch.FloatTensor(images).cuda())
#         targets = Variable(torch.FloatTensor(targets).cuda())

#         optimizer.zero_grad()
#         preds = model(images)
#         loss = criterion(preds, targets)
#         loss.backward()
#         losses.append(loss.item())
#         print(loss.item())

#         optimizer.step()

#     e_loss = np.mean(losses)
#     scheduler.step(e_loss)



# # Training

# In[31]:


def train_last_layer(learn, loss_recorder, n_cycle, lr=1e-4):
    print(f'Train last layer group with {n_cycle} cycle, lr={lr}')
    learn.freeze_to(-1)
    md = get_md(224, 128, percentage)
    learn.set_data(md)
    layer_opt = learn.get_layer_opt(lr, None)
    loss_recorder.new_phase(layer_opt)
    learn.sched = None
    learn.fit_gen(learn.model, learn.data, layer_opt, n_cycle,                   metrics=[auc_s, acc_s], cycle_len=1, use_clr=(8, 10),                   callbacks=[loss_recorder], best_save_name='best')
#     print()
#     plt.figure()
#     loss_recorder.plot_loss()
#     plt.figure()
#     learn.sched.plot_lr()

def train_all_layer(learn, loss_recorder, n_cycle, lr=8e-3):
    print(f'Train every groups with {n_cycle} cycle, lr={lr}')
    lrs = [lr/1000, lr/100, lr]
    learn.unfreeze()
    md = get_md(224, 64, percentage)
    learn.set_data(md)
    layer_opt = learn.get_layer_opt(lrs, None)
    loss_recorder.new_phase(layer_opt)
    learn.sched = None
    learn.fit_gen(learn.model, learn.data, layer_opt, n_cycle,                   metrics=[auc_s, acc_s], cycle_len=1, use_clr=(8, 10),                   callbacks=[loss_recorder], best_save_name='best')
#     print()
#     plt.figure()
#     loss_recorder.plot_loss()
#     plt.figure()
#     learn.sched.plot_lr()


# In[32]:


loss_recorder = TrainingRecoder(model_path)
train_last_layer(learn, loss_recorder, 50, lr=1e-3) # warm up
train_all_layer(learn, loss_recorder, 50, lr=1e-4)
# train_all_layer(learn, loss_recorder, 2, lr=8e-3) # super convergence
# train_all_layer(learn, loss_recorder, 1, lr=8e-2) # super convergence
# train_all_layer(learn, loss_recorder, 10, lr=4e-4) # slow down
# train_last_layer(learn, loss_recorder, 10, lr=4e-4) # settle down
loss_recorder.save()


# In[42]:


# loss_recorder.plot_loss()
# plt.figure()
# loss_recorder.plot_lr()
# plt.figure()
# plt.plot(loss_recorder.rec_metrics)
# plt.plot(np.stack(loss_recorder.rec_metrics)[:, 0])
# plt.plot(loss_recorder.trn_losses)
# plt.plot(loss_recorder.val_losses)


# In[17]:


def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))

print('Test')
py, y = learn.TTA()
print(auc_np(sigmoid_np(np.mean(py, axis=0)), y))


# In[ ]:


# lr = 1e-4
# learn.fit(lr, 1)


# ## Class Activation Map

# In[ ]:


# x, y = next(iter(md.val_dl))
# x, y = x[None, 1], y[None, 1]


# In[33]:


# class FeatureHook:
#     features = None

#     def __init__(self, m):
#         self.hook = m.register_forward_hook(self.hook_fn)

#     def hook_fn(self, module, input, output):
#         self.features = output

#     def remove(self):
#         self.hook.remove()

# class WeightHook:
#     weights = None

#     def __init__(self, m):
#         self.hook = m.register_forward_hook(self.hook_fn)

#     def hook_fn(self, module, input, output):
#         self.weights = list(module.parameters())[0].data

#     def remove(self):
#         self.hook.remove()

# last_conv_layer = learn[-2][-1]
# linear_layer = learn[-1][2]
# feat = FeatureHook(last_conv_layer)
# weig = WeightHook(linear_layer)

# py = learn.model(x)

# feat.remove()
# weig.remove()


# In[64]:


# learn[-2][-1]


# In[42]:


# feat.features.shape, weig.weights.shape


# In[46]:


# feat.features[0]
# i = y.argmax()
# f2 = np.dot(np.rollaxis(to_np(feat.features[0]), 0, 3), to_np(weig.weights[i]))
# f2 -= f2.min()
# f2 /= f2.max()
# # get the CAM for correct class
# plt.imshow(md.val_ds.denorm(x)[0])
# plt.imshow(cv2.resize(f2, (224, 224)), alpha=0.5, cmap='hot')


# In[29]:


# %who FeatureHook


# In[32]:


# %reset_selective -f weig


# ## Activation Heat Map

# In[62]:


# H = torch.max(feat.features[0], dim=0)
# plt.imshow(md.val_ds.denorm(x)[0])
# plt.imshow(cv2.resize(to_np(H[0]), (224, 224)), alpha=0.5, cmap='hot')
# feat.features[0]


# In[60]:


# theta = .7
# slice_x, slice_y = ndimage.find_objects(h[0] > theta, True)[0]
# H[0]
# rect = Rectangle(xy, width, height, linewidth=1,edgecolor='r',facecolor='none')

# # plt.imshow(, (224, 224)))


# ## Knn for similar image

# ## Focal loss
