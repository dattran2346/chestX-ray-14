import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE_IDS = [0]
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(current_dir, os.path.pardir))

N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']
EPOCHS = 100
BATCHES = 500
BATCHSIZE = 32 #64*2
VALIDATE_EVERY_N_EPOCHS = 5
SCALE_FACTOR = .875
DATA_DIR = '/home/dattran/data/xray/'
PERCENTAGE = 0.01 # percentage of data use for quick run
TEST_AGUMENTED = True
DISEASE_THRESHOLD = 0.5

MODEL_DIR = '%s/models' % ROOT
LOG_DIR = '%s/logs' % ROOT
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
