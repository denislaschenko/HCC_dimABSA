import os

#Task settings
ACTIVE_SUBTASK = "subtask_3"
LANG = "eng"
# DOMAIN = os.getenv("DOMAIN", "laptop")
DOMAIN = "restaurant"
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
        "RAG_K": 3
    },
   "subtask_3": {
        "SUBTASK": "subtask_3",
        "TASK": "task3",
        "INCLUDE_OPINION": True,
        "INCLUDE_CATEGORY": True,
        "MODEL_NAME": "princeton-nlp/sup-simcse-roberta-large",
        "PATIENCE": 2,
        "RAG_K": 3
    }
}


# Model settings
MODEL_NAME = "FacebookAI/roberta-large"
MODEL_VERSION_ID = "v1.4"

# Training settings
SEED = 100
LEARNING_RATE = 1e-5 # switched from 2e-5
EPOCHS = 20
BATCH_SIZE = 128 # switched from 8
MAX_LEN = 128
PATIENCE = 3
NUM_BINS = 9
SIGMA = 1

NUM_WORKERS = 4
ENSEMBLE_SEEDS = [100, 42, 2026]

current_config = TASK_CONFIGS[ACTIVE_SUBTASK]
globals().update(current_config)

#Data file paths
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONFIG_DIR, '..', '..'))
DATA_ROOT = "task-dataset"
TRACK = "track_a"


# Data file Path Constructor
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_ROOT, TRACK, ACTIVE_SUBTASK, LANG)
FIGURE_DIR = os.path.join(PROJECT_ROOT, "outputs/subtask_1/figures")
CSV_FILE = os.path.join(PROJECT_ROOT, "outputs/subtask_1/logs/results.csv")

TRAIN_FILE = os.path.join(PROJECT_ROOT, DATA_ROOT, TRACK, ACTIVE_SUBTASK, LANG, f"{LANG}_{DOMAIN}_train_FULL.jsonl")
# TRAIN_FILE = os.path.join(PROJECT_ROOT, DATA_ROOT, TRACK, ACTIVE_SUBTASK, LANG, f"{LANG}_{DOMAIN}_train_SMALL.jsonl")
PREDICT_FILE = os.path.join(DATA_DIR, f"{LANG}_{DOMAIN}_test_{current_config.get('TASK')}.jsonl")

# Output settings
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "outputs", current_config.get("SUBTASK"), "models")
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f"best_model_{DOMAIN}.pt")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", current_config.get("SUBTASK"), "predictions")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"pred_{LANG}_{DOMAIN}.jsonl")

if __name__ == '__main__':
    print(DATA_DIR)
