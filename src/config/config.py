import os


# Data directory contains the folders with different classes.
DATA_DIR_4 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "data")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "nn_models/logs")
CHKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "nn_models/tf_ckpt")
UPSAMPLE_DIR_EXTENSION = '_upsample'
UPSAMPLE_NAME = 'upsample'
DOWNSAMPLE_NAME = 'downsample'
ORIGINAL_NAME = 'original'


TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2


