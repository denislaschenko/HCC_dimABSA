import os
MODEL_NAME = "bert-base-uncased"

MAX_LEN = 128
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 10
PATIENCE = 2

TRAIN_FILE = "task-dataset/track_a/subtask_1/eng/eng_laptop_train_alltasks.jsonl"
DEV_FILE   = "task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl"
TEST_FILE  = "data/subtask2/test.jsonl"
PRED_OUTPUT_FILE = "data/subtask2/test_output.jsonl"

PROCESSED_TRAIN = "data/subtask2/train_bio.jsonl"
PROCESSED_DEV   = "data/subtask2/dev_bio.jsonl"
PROCESSED_TEST  = "data/subtask2/test_bio.jsonl"

MODEL_SAVE_PATH = "models/subtask2/best_model.pt"
