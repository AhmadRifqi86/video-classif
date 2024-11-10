import os
import torch

DATASET_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/train'  # Path to dataset
VAL_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/val'
TEST_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/test'
PROCESSED_DATA_PATH = "/home/arifadh/Desktop/Skripsi-Magang-Proyek/temporary"
DATA_FILE = os.path.join(PROCESSED_DATA_PATH, "X_data_700.npy")
LABELS_FILE = os.path.join(PROCESSED_DATA_PATH, "y_labels_700.npy")
CLASSES_FILE = os.path.join(PROCESSED_DATA_PATH, "class_labels_700.pkl")
TEST_PATH = '/path/to/test'
IMG_HEIGHT, IMG_WIDTH = 80, 80 # Image dimensions
SEQUENCE_LENGTH = 40
BATCH_SIZE = 2
HIDDEN_SIZE = 56
CNN_BACKBONE = "resnet50"
RNN_INPUT_SIZE = 512
RNN_LAYER = 2
RNN_TYPE = "lstm"
SAMPLING_METHOD = "uniform"
RNN_OUT = "all"
MAX_VIDEOS = 700
EPOCH = 30
FINETUNE = True
CLASSIF_MODE = "multiclass"
MODEL_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/model.pth'  # Path to save model
EARLY_STOP = 0.0

# Transfer configuration to variables
CONF_SEQUENCE_LENGTH = SEQUENCE_LENGTH
CONF_BATCH_SIZE = BATCH_SIZE
CONF_HIDDEN_SIZE = HIDDEN_SIZE
CONF_CNN_BACKBONE = CNN_BACKBONE
CONF_RNN_INPUT_SIZE = RNN_INPUT_SIZE
CONF_RNN_LAYER = RNN_LAYER
CONF_RNN_TYPE = RNN_TYPE
CONF_SAMPLING_METHOD = SAMPLING_METHOD
CONF_RNN_OUT = RNN_OUT
CONF_MAX_VIDEOS = MAX_VIDEOS
CONF_EPOCH = EPOCH
CONF_FINETUNE = FINETUNE
CONF_MODEL_PATH = MODEL_PATH
CONF_CLASSIF_MODE = CLASSIF_MODE
CONF_EARLY_STOP = EARLY_STOP
CONF_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")