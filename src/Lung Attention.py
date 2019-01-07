
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


# In[3]:


PATH = Path('/mnt/data/xray-thesis/data/chestX-ray14')
IMAGE_DN = 'images'
TRAIN_CSV = 'train_list.csv'
TEST_CSV = 'test_list.csv'
sz = 224
bs = 64
arch = resnet50
N_CLASSES = 15


# ## Load dataset

# In[4]:


def aucs(targets, preds):
    aurocs = []
    for i in range(N_CLASSES):
        aurocs.append(roc_auc_score(targets[:, i], preds[:, i]))
    return aurocs


# In[5]:


aug_tmfs = [RandomFlip(),
            RandomCrop(sz)]
tmfs = tfms_from_model(arch, sz, aug_tmfs, 1.05)
md = ImageClassifierData.from_csv(PATH, 'images', PATH/TRAIN_CSV, bs, tmfs, cat_separator='|')


# In[6]:


x, y = next(iter(md.val_dl))


# ## Densenet121 arch with binary classification

# In[6]:


class LSEPool2d(nn.Module):
    
    def __init__(self, r=3):
        super().__init__()
        self.r =r
    
    def forward(self, x):
        s = x.size()[3]  # x: bs*2048*7*7
        r = self.r
        x_max = F.adaptive_max_pool2d(x, 1) # x_max: bs*2048*1*1
        p = ((1/r) * torch.log((1 / (s*s)) * torch.exp(r*(x - x_max)).sum(3).sum(2)))
        x_max = x_max.view(x.size(0), -1) # bs*2048
        return x_max+p
        
        
head = nn.Sequential(LSEPool2d(), 
                     Flatten(),
                     nn.Linear(2048, 15),
                     nn.Sigmoid())


# In[7]:


models = ConvnetBuilder(arch, 0, 0, 0, custom_head=head)
learn = ConvLearner(md, models)
learn.crit = nn.BCELoss()


# In[8]:


learn.summary()


# In[71]:


learn.freeze_to(-1)
learn.lr_find()


# In[66]:


learn.sched.plot(0, 0)


# In[72]:


lr = 1e-2
learn.fit(lr, 3)


# In[67]:


# py, y = learn.TTA()


# In[75]:


# y.shape, .shape
aucs = aucs(y, np.mean(py, axis=0))
np.mean(aucs)


# In[92]:


learn.save('tmp')


# In[8]:


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
