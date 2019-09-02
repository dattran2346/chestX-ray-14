import os
from pathlib import Path
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(current_dir, os.path.pardir))

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
                'Fibrosis', 'Pleural Thickening', 'Hernia']

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

PATH = Path('/home/dattran/data/xray-thesis/chestX-ray14')
ATTENTION_DN = 'tmp/attention'
IMAGE_DN = 'images'
TRAIN_CSV = 'train.csv'
VAL_CSV = 'val.csv'
TEST_CSV = 'test.csv'

"""
Below may not need any more
"""
# EPOCHS = 2# 100
# # BATCHES = 500 # 500
# BATCHSIZE = 32
# VALIDATE_EVERY_N_EPOCHS = 5
SCALE_FACTOR = .875
DATA_DIR = '/mnt/data/xray-thesis/data/chestX-ray14/images/'
PERCENTAGE = 0.01 # percentage of data use for quick run
TEST_AGUMENTED = False
DISEASE_THRESHOLD = 0.5

MODEL_DIR = '/mnt/data/xray-thesis/models'
LOG_DIR = 'mnt/data/xray-thesis/logs'
CSV_DIR = '%s/csv' % ROOT
STAT_DIR = '%s/stats' % ROOT

# chexnet file
CHEXNET_MODEL_NAME = '%s/chexnet_densenet.pth.tar' % MODEL_DIR
CHEXNET_TRAIN_CSV = '%s/chexnet_train_list.csv' % CSV_DIR
CHEXNET_VAL_CSV = '%s/chexnet_val_list.csv' % CSV_DIR
CHEXNET_TEST_CSV = '%s/chexnet_test_list.csv' % CSV_DIR
TRAIN_CSV = '%s/train_list.csv' % CSV_DIR
VAL_CSV = '%s/val_list.csv' % CSV_DIR
TEST_CSV = '%s/test_list.csv' % CSV_DIR

# different model
DENSENET121_DIR = '%s/densenet121' % MODEL_DIR

# stat
TRAIN_STAT = '%s/train.csv' % STAT_DIR
TEST_STAT = '%s/test.csv' % STAT_DIR

PREPROCESS = False
