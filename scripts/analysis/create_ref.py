import json
import random

from src.shared import config

with open(config.TRAIN_FILE,'r',encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

random.seed(config.SEED)
random.shuffle(data)
split_idx = int(len(data)*0.9)

train_split = data[:split_idx]
dev_split = data[split_idx:]

with open("../../task-dataset/track_a/subtask_3/eng/local/local_train.jsonl", 'w', encoding='utf-8') as f:
    for item in train_split:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

dev_input = []
for item in dev_split:
    dev_input.append({"ID": item["ID"],"Text": item["Text"]})

with open("../../task-dataset/track_a/subtask_3/eng/local/local_dev_input.jsonl", 'w', encoding='utf-8') as f:
    for item in dev_input:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("../../task-dataset/track_a/subtask_3/eng/local/local_dev_gold.jsonl", 'w', encoding='utf-8') as f:
    for item in dev_split:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Created local_train.jsonl ({len(train_split)}) and local_dev_input.jsonl ({len(dev_input)})")
