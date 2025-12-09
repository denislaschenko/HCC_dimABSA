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
NUM_BINS = 81

#Data file paths
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONFIG_DIR, '..', '..'))
DATA_ROOT = "task-dataset"
TRACK = "track_a"


# Data file Path Constructor
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_ROOT, TRACK, SUBTASK, LANG)
TRAIN_FILE = os.path.join(DATA_DIR, f"{LANG}_{DOMAIN}_train_alltasks.jsonl")
PREDICT_FILE = os.path.join(DATA_DIR, f"{LANG}_{DOMAIN}_dev_{TASK}.jsonl")

# Output settings
MODEL_SAVE_DIR = os.path.join("outputs", "subtask_1", "models")
PREDICTION_DIR = os.path.join("outputs", "subtask_1", "predictions")


PREDICTION_FILE = os.path.join(PREDICTION_DIR, f"pred_{LANG}_{DOMAIN}.jsonl")
TEST_FILE = os.path.join(PREDICTION_DIR, f"test_{LANG}_{DOMAIN}.jsonl")

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, MODEL_SAVE_DIR, SUBTASK, "best_model.pt")
