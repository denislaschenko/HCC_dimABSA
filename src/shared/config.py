import os

#Task settings
ACTIVE_SUBTASK = "subtask_3"
LANG = "eng"
DOMAIN = "laptop"

TASK_CONFIGS = {
    "subtask_1": {
        "SUBTASK": "subtask_1",
        "TASK": "task1",
        "INCLUDE_OPINION": False,
        "MODEL_NAME": "FacebookAI/roberta-large",
        "PATIENCE": 2,
    },
    "subtask_2": {
        "SUBTASK": "subtask_2",
        "TASK": "task2",
        "INCLUDE_OPINION": True,
        "MODEL_NAME": "FacebookAI/roberta-large",
        "PATIENCE": 10,
    },
   "subtask_3": {
        "SUBTASK": "subtask_3",
        "TASK": "task3",
        "INCLUDE_OPINION": True,
        "INCLUDE_CATEGORY": True,
        "MODEL_NAME": "roberta-base",
        "PATIENCE": 2,
    }
}


# Model settings
MODEL_NAME = "roberta-base"
MODEL_VERSION_ID = "v1.4"

# Training settings
SEED = 100
LEARNING_RATE = 2e-5
EPOCHS = 20
BATCH_SIZE = 8
MAX_LEN = 128
PATIENCE = 3
NUM_BINS = 9
SIGMA = 1

current_config = TASK_CONFIGS[ACTIVE_SUBTASK]
globals().update(current_config)

#Data file paths
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONFIG_DIR, '..', '..'))
DATA_ROOT = "task-dataset"
TRACK = "track_a"


# Data file Path Constructor
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_ROOT, TRACK, ACTIVE_SUBTASK, LANG)
CSV_DIR = os.path.join(PROJECT_ROOT, "outputs/subtask_1/logs/results.csv")
TRAIN_FILE = os.path.join(PROJECT_ROOT, DATA_ROOT, TRACK, "subtask_1", LANG, f"eng_{DOMAIN}_production_train.jsonl")
PREDICT_FILE = os.path.join(DATA_DIR, f"{LANG}_{DOMAIN}_test_{current_config.get('TASK')}.jsonl")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "outputs/subtask_1/figures")

# Output settings
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "outputs", current_config.get("SUBTASK"), "models")
PREDICTION_DIR = os.path.join(PROJECT_ROOT, "outputs", current_config.get("SUBTASK"), "predictions")


PREDICTION_FILE = os.path.join(PREDICTION_DIR, f"pred_{LANG}_{DOMAIN}.jsonl")
TEST_FILE = os.path.join(PREDICTION_DIR, f"test_{LANG}_{DOMAIN}.jsonl")

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, ACTIVE_SUBTASK, f"best_model_{DOMAIN}.pt")

if __name__ == '__main__':
    print(DATA_DIR)
