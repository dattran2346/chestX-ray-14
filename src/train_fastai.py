from fastai.conv_learner import *
from fastai.dataset import *
from fastai.transforms import *
from fastai.models.resnet import vgg_resnet50
from sklearn.metrics import roc_auc_score
import pixiedust
from matplotlib.patches import Rectangle
from datetime import datetime
from layers import *
from utils import aucs

PATH = Path('/mnt/data/xray-thesis/data/chestX-ray14')
IMAGE_DN = 'images'
TRAIN_CSV = 'train_list.csv'x
TEST_CSV = 'test_list.csv'
arch = densenet121
N_CLASSES = 15
model_name = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

def get_data(sz, bs):
    aug_tmfs = [RandomScale(sz, sz*1.5),
                RandomFlip(),
                RandomCrop(sz)]
    tmfs = tfms_from_model(arch, sz, aug_tmfs)
    md = ImageClassifierData.from_csv(PATH, 'images', PATH/TRAIN_CSV, bs, tmfs, cat_separator='|')
    return md

md = get_data(sz=64, bs=64)

# construct model
densenet_head = nn.Sequential(LSEPool2d(1), 
                     nn.Linear(1024, 15))
densenet_model = ConvnetBuilder(arch, 0, 0, 0, xtra_cut=1, custom_head=densenet_head)
learn = ConvLearner(md, densenet_model)
learn.crit = WeightedBCELoss()

lr = 1e-4
wd=1e-7
lrs = np.array([lr/100,lr/10,lr])

# train
learn.freeze_to(-1)
learn.set_data(get_data(sz=64, bs=64)) 
learn.fit(lr,1,wds=wd,cycle_len=8,use_clr=(5,8))

learn.unfreeze()
learn.bn_freeze(True)
learn.fit(lrs, 1, wds=wd, cycle_len=4,use_clr=(5, 8)) # 1.5GB

# learn.save('')
