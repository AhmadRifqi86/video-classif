import os
import torch

APP_STAGE = os.getenv("APP_STAGE", "devel")

DATASET_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/train'  # Path to dataset
VAL_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/val'
TEST_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/test'
PROCESSED_DATA_PATH = "/home/arifadh/Desktop/Skripsi-Magang-Proyek/temporary"
TEST_PATH = '/path/to/test'
IMG_HEIGHT, IMG_WIDTH = 80, 80 # Image dimensions
SEQUENCE_LENGTH = 60
BATCH_SIZE = 16
HIDDEN_SIZE = 25
CNN_BACKBONE = "resnet34"
RNN_INPUT_SIZE = 23
RNN_LAYER = 2
RNN_TYPE = "mamba"
SAMPLING_METHOD = "uniform"
RNN_OUT = "all"
MAX_VIDEOS = 1000
EPOCH = 8
DROPOUT = 0.3
FINETUNE = True
BIDIR = False
CLASSIF_MODE = "multiclass"
MODEL_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/model.pth'  # Path to save model
EARLY_STOP = 0.0
DATA_FILE = os.path.join(PROCESSED_DATA_PATH, f"X_data_{MAX_VIDEOS}_{SEQUENCE_LENGTH}fr_{SAMPLING_METHOD}.npy")
LABELS_FILE = os.path.join(PROCESSED_DATA_PATH, f"y_labels_{MAX_VIDEOS}_{SEQUENCE_LENGTH}fr_{SAMPLING_METHOD}.npy")
#CLASSES_FILE = os.path.join(PROCESSED_DATA_PATH, f"class_labels_{MAX_VIDEOS}_{SEQUENCE_LENGTH}fr_{SAMPLING_METHOD}.pkl")
CLASSES_FILE = os.path.join(PROCESSED_DATA_PATH, f"class_labels_{MAX_VIDEOS}_{SEQUENCE_LENGTH}fr_{SAMPLING_METHOD}.pkl.npy")


#automation, deployment, data collection
CONFIG_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/medsos_lrcn/src/all_config.py'
SOURCE_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/medsos_lrcn/src/main.py'  #ini nanti ganti nama 
LOG_FILE_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/medsos_lrcn/src/new_medsos_log_genetic.txt'
BEST_MODEL_DIR = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/best_models_medsos_genetic/'
TEST_RUNS = 3  # Number of times to test each configuration
CHECKPOINT_FILE = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/medsos_lrcn/src/genetic_medsos_checkpoint.json'  # File to track best results
SLEEP = 60
VIDEO_DIR = '/home/arifadh/Downloads/tiktok_videos/'
BACKEND_URL = "http://backend:5000/classify" if APP_STAGE == "prod" else "http://localhost:5000/classify"  #harus mindahin ini ke all_config
BACKEND_CHECKER = "http://backend:5000/video_labels" if APP_STAGE == "prod" else "http://localhost:5000/video_labels"  #harus mindahin ini ke all_config
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "video_classification"
COLLECTION_NAME = "classification_results"

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
CONF_DROPOUT = DROPOUT
CONF_FINETUNE = FINETUNE
CONF_MODEL_PATH = MODEL_PATH
CONF_CLASSIF_MODE = CLASSIF_MODE
CONF_EARLY_STOP = EARLY_STOP
CONF_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_BIDIR = BIDIR


# first best
# IMG_HEIGHT, IMG_WIDTH = 80, 80 # Image dimensions
# SEQUENCE_LENGTH = 40
# BATCH_SIZE = 4
# HIDDEN_SIZE = 16  #best 16
# CNN_BACKBONE = "densenet121"
# RNN_INPUT_SIZE = 128 atau 96 #best 256
# RNN_LAYER = 2
# RNN_TYPE = "mamba"
# SAMPLING_METHOD = "uniform"
# RNN_OUT = "all"
# MAX_VIDEOS = 700
# EPOCH = 3


#second
# IMG_HEIGHT, IMG_WIDTH = 80, 80 # Image dimensions
# SEQUENCE_LENGTH = 40
# BATCH_SIZE = 4
# HIDDEN_SIZE = 16  #best 16
# CNN_BACKBONE = "densenet121"
# RNN_INPUT_SIZE = 96  #best 256
# RNN_LAYER = 3
# RNN_TYPE = "mamba"
# SAMPLING_METHOD = "uniform"
# RNN_OUT = "all"
# MAX_VIDEOS = 700
# EPOCH = 3
# FINETUNE = True
# CLASSIF_MODE = "multiclass"
# MODEL_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/model.pth'  # Path to save model
# EARLY_STOP = 0.0



# IMG_HEIGHT, IMG_WIDTH = 80, 80 # Image dimensions
# SEQUENCE_LENGTH = 40
# BATCH_SIZE = 8
# HIDDEN_SIZE = 16,32  #best 16
# CNN_BACKBONE = "resnet34" #best resnet50, resnet101
# RNN_INPUT_SIZE = 16,24  #best 16
# RNN_LAYER = 2
# RNN_TYPE = "mamba"
# SAMPLING_METHOD = "uniform"
# RNN_OUT = "all"
# MAX_VIDEOS = 700
# EPOCH = 8,6