
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from fastai.conv_learner import *
from fastai.dataset import *
from fastai.transforms import *
from fastai.models.resnet import vgg_resnet50
from sklearn.metrics import roc_auc_score
import pixiedust
from matplotlib.patches import Rectangle
from datetime import datetime
import pandas as pd
from constant import CLASS_NAMES


# In[3]:


PATH = Path('/mnt/data/xray-thesis/data/chestX-ray14')
IMAGE_DN = 'images'
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
TEST_CSV = 'test.csv'

arch = densenet121

sz = 128
bs = 32

N_CLASSES = 14
model_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
model_name


# ## Load dataset

# In[5]:


class ChestXray14Dataset(FilesDataset):
    '''
    Get image for train, validate and test base on NIH split
    '''

    def __init__(self, image_names, labels, transform, path, percentage=0.1):
        self.labels = labels
        self.percentage = percentage
        super().__init__(image_names, transform, path)
        
        
    def get_y(self, i):
        return self.labels[i]
    
    def get_c(self):
        return 14

    def get_n(self):
        return int(self.percentage*super().get_n())

trn_df = pd.read_csv(PATH/TRAIN_CSV, sep=' ', header=None)
trn_images = trn_df.iloc[:, 0].values
trn_labels = trn_df.iloc[:, 1:].values
val_df = pd.read_csv(PATH/VAL_CSV, sep=' ', header=None)
val_images = val_df.iloc[:, 0].values
val_labels = val_df.iloc[:, 1:].values

aug_tmfs = [RandomScale(sz, sz*1.5),
            RandomFlip(),
            RandomCrop(sz)]
tfms = tfms_from_model(arch, sz, aug_tmfs)


datasets = ImageData.get_ds(ChestXray14Dataset, (trn_images, trn_labels), (val_images, val_labels), tfms, path=PATH/IMAGE_DN)
md = ImageData(PATH/IMAGE_DN, datasets, bs, 8, CLASS_NAMES)


# In[6]:


x, y = next(iter(md.val_dl))


# In[2]:


# %pixie_debugger
# def get_data(sz, bs):
#     # TODO: Get ImageData ChestXray
#     md = ImageClassifierData.from_csv(PATH, 'images', PATH/TRAIN_CSV, bs, tmfs, cat_separator='|')
#     return md


# ## Densenet121 arch with binary classification

# In[1]:


# class FocalLoss(WeightedBCELoss):
    
#     def __init__(self, theta=2):
#         super().__init__()
#         self.theta = theta
        
#     def forward(self, input, target):
# #         pt = target*input + (1-target)*(1-input)
# #         target *= (1-pt)**self.theta
#         w = self.get_weight(input, target)
#         return F.binary_cross_entropy_with_logits(input, target, w)



# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()


# In[59]:


class ChexNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.backbone = densenet121(True).features
        self.head = nn.Sequential(LSEPool2d(1), 
                                  nn.Linear(1024, N_CLASSES))
    
    def forward(self, x):
        return self.head(self.backbone(x))

class CheXNetModel():
    
    def __init__(self, chexnet):
        self.model = chexnet
    
    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.backbone), [8]))
        lgs += [children(self.model.head)]
        return lgs

# chexnet.parameters is freezed except the last 2 layer
chexnet = ChexNet()
chexnet_model = CheXNetModel(chexnet)
lgs = model.get_layer_groups(True)
learn = ConvLearner(md, chexnet_model)


# In[ ]:


learn.lr_find()


# In[13]:


learn.sched.plot(0, 700)


# In[72]:


# lr = 1e-
# learn.fit(lr, 3)


# In[2]:


py, y = learn.TTA()
aucs = aucs(y, np.mean(py, axis=0))
np.mean(aucs)


# In[ ]:


lr = 1e-4
learn.set_data(get_data(128, 32)) # 64, 64: 1GB; 128, 64: 4GB
learn.fit(lr, 1)


# In[92]:


learn.save('tmp')


# In[10]:


learn.load('tmp')


# ## Class Activation Map

# In[ ]:


x, y = next(iter(md.val_dl))
x, y = x[None, 1], y[None, 1]


# In[33]:


class FeatureHook:
    features = None
    
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output
        
    def remove(self):
        self.hook.remove()
        
class WeightHook:
    weights = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.weights = list(module.parameters())[0].data
    
    def remove(self):
        self.hook.remove()

last_conv_layer = learn[-2][-1]
linear_layer = learn[-1][2]
feat = FeatureHook(last_conv_layer)
weig = WeightHook(linear_layer)

py = learn.model(x)

feat.remove()
weig.remove()


# In[64]:


learn[-2][-1]


# In[42]:


feat.features.shape, weig.weights.shape


# In[46]:


feat.features[0]
i = y.argmax()
f2 = np.dot(np.rollaxis(to_np(feat.features[0]), 0, 3), to_np(weig.weights[i]))
f2 -= f2.min()
f2 /= f2.max()
# get the CAM for correct class
plt.imshow(md.val_ds.denorm(x)[0])
plt.imshow(cv2.resize(f2, (224, 224)), alpha=0.5, cmap='hot')


# In[29]:


get_ipython().run_line_magic('who', 'FeatureHook')


# In[32]:


get_ipython().run_line_magic('reset_selective', '-f weig')


# ## Activation Heat Map

# In[62]:


H = torch.max(feat.features[0], dim=0)
plt.imshow(md.val_ds.denorm(x)[0])
plt.imshow(cv2.resize(to_np(H[0]), (224, 224)), alpha=0.5, cmap='hot')
feat.features[0]


# In[60]:


theta = .7
slice_x, slice_y = ndimage.find_objects(h[0] > theta, True)[0]
H[0]
rect = Rectangle(xy, width, height, linewidth=1,edgecolor='r',facecolor='none')

# plt.imshow(, (224, 224)))


# ## Knn for similar image

# ## Focal loss
