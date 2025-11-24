import os

# Task settings
SUBTASK = "subtask_1"
TASK = "task1"
LANG = "eng"
DOMAIN = "laptop"

# Model settings
MODEL_NAME = "roberta-base"
MODEL_VERSION_ID = "v1.4"

# Training settings
LEARNING_RATE = 1.1796630496108734e-05
EPOCHS = 20
BATCH_SIZE = 4
MAX_LEN = 128
PATIENCE = 2

#Data file paths
DATA_ROOT = "task-dataset"
TRACK = "track_a"

# Data file Path Constructor
DATA_DIR = f"{DATA_ROOT}/{TRACK}/{SUBTASK}/{LANG}"
TRAIN_FILE = f"{DATA_DIR}/{LANG}_{DOMAIN}_train_alltasks.jsonl"
PREDICT_FILE = f"{DATA_DIR}/{LANG}_{DOMAIN}_dev_{TASK}.jsonl"

# Output settings
MODEL_SAVE_DIR = "models"
PREDICTION_DIR = "predictions"


PREDICTION_FILE = f"{PREDICTION_DIR}/{SUBTASK}/pred_{LANG}_{DOMAIN}.jsonl"

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, SUBTASK, "best_model.pt")

