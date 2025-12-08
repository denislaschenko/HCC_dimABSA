import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.shared import utils
from src.subtask_1 import config
from src.subtask_1.dataset import VADataset
from src.subtask_1.model import TransformerVARegressor

model_name = config.MODEL_NAME
dropout = 0.1
model_path = config.MODEL_SAVE_PATH
batch_size = config.BATCH_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from {model_path}...")
model = TransformerVARegressor(model_name, dropout)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

print(f"Loading training data from {config.TRAIN_FILE}...")
raw_train = utils.load_jsonl(config.TRAIN_FILE)
df = utils.jsonl_to_df(raw_train)

print("Fixing Aspect column format...")
df["Aspect"] = df["Aspect"].apply(lambda x: [x] if isinstance(x, str) else x)

utils.set_seed(25)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = VADataset(df, tokenizer, max_len=config.MAX_LEN)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print("Running inference on full training set...")
pred_v, pred_a, gold_v, gold_a = utils.get_predictions(model, train_loader, device, type="dev")

df["Predicted_Valence"] = pred_v
df["Predicted_Arousal"] = pred_a

df["Resid_Valence"] = df["Valence"] - df["Predicted_Valence"]
df["Resid_Arousal"] = df["Arousal"] - df["Predicted_Arousal"]

output_file = os.path.join(config.PREDICTION_SUBDIR, "train_set_residual_analysis.jsonl")
print(f"Saving results to {output_file}...")

df.to_json(output_file, orient='records', lines=True)

print("Done.")

