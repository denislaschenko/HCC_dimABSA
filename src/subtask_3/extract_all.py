import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from unicodedata import category

from src.shared.config import PREDICTION_FILE

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared import config, utils
from src.subtask_2 import SFT_Slop2

def load_training_data(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item.get("Text")
                quads = item.get("Quadruplet", [])


                for q in quads:
                    aspect = q.get("Aspect")
                    cat = q.get("Category")
                    opinion = q.get("Opinion")

                    if cat:
                        data.append({"Text": text, "Aspect": aspect, "Category": cat, "Opinion": opinion})
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return []

    return data


def load_inference_data(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item.get("Text")
                item_id = str(item.get("ID"))

                triplets = item.get("Triplet", [])

                if not triplets:
                    continue

                for t in triplets:
                    data.append({
                        "ID": item_id,
                        "Text": text,
                        "Aspect": t.get("Aspect"),
                        "Opinion": t.get("Opinion")
                    })

    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
    return data


class CategoryDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len, label_encoder=None, is_inference=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data_list
        self.is_inference = is_inference
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        text = item.get("Text", "")
        aspect = item.get("Aspect", "")
        opinion = item.get("Opinion", "")

        input_text = f"{aspect} {self.tokenizer.sep_token} {opinion} {self.tokenizer.sep_token} {text}"

        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        sample = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }

        if not self.is_inference:
            label_str = item["Category"]
            label_id = self.label_encoder.transform([label_str])[0]
            sample["labels"] = torch.tensor(label_id, dtype=torch.long)
        else:
            sample["ID"] = item["ID"]
            sample["Aspect"] = item["Aspect"]

        return sample

class ContrastiveCategoryModel(torch.nn.Module):
    def __init__(self, model_name):
        super(ContrastiveCategoryModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(cls_embedding, dim=1)

def main():
    start_time = time.time()
    subtask2_pred_file = os.path.join(config.PROJECT_ROOT, "outputs", "subtask_2", "predictions",
                                      f"pred_{config.LANG}_{config.DOMAIN}.jsonl")
    # SFT_Slop2.main()
    print("\n--- Step 1: Running Integrated Triplet Inference ---")
    import subprocess

    adapter_path = "/workspace/HCC_dimABSA_remote/outputs/subtask_2/models/final_adapter"
    inference_script = os.path.join(config.PROJECT_ROOT, "src", "subtask_2", "inference.py")

    subprocess.run([
        sys.executable,
        inference_script,
        "--checkpoint", adapter_path
    ], check=True)

    print("\n--- Step 2: Proceeding to Subtask 3 Category Extraction ---")

    if not os.path.exists(subtask2_pred_file):
        raise FileNotFoundError(f"Subtask 2 predictions not found at {subtask2_pred_file}. Run inference.py first.")

    target_file = PREDICTION_FILE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Training Data (Memory Bank)...")

    # # debug train data for speedtest
    train_data = load_training_data(config.TRAIN_FILE)
    # train_data = train_data[:15]

    all_categories = sorted(list(set(d["Category"] for d in train_data)))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_categories)
    print(f"Found {len(all_categories)} categories.")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_dataset = CategoryDataset(train_data, tokenizer, config.MAX_LEN, label_encoder, is_inference=False)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)



    model = ContrastiveCategoryModel(config.MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = utils.SupConLoss(temperature=0.07)

    print("\n--- Training Contrastive Encoder ---")
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            features = model(input_ids, mask)

            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

    print("\n--- Generating Embeddings for Inference ---")
    model.eval()

    memory_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    bank_embeddings = []
    bank_labels = []

    with torch.no_grad():
        for batch in tqdm(memory_loader, desc="Encoding Memory Bank"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            emb = model(input_ids, mask)
            bank_embeddings.append(emb.cpu().numpy())
            bank_labels.extend(batch["labels"].cpu().numpy())

    X_train = np.vstack(bank_embeddings)
    y_train = np.array(bank_labels)

    print("Fitting KNN Classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance')
    knn.fit(X_train, y_train)

    print(f"Loading Prediction Data: {subtask2_pred_file}")
    pred_data = load_inference_data(subtask2_pred_file)
    pred_dataset = CategoryDataset(pred_data, tokenizer, config.MAX_LEN, is_inference=True)
    pred_loader = DataLoader(pred_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    results = {}

    with torch.no_grad():
        for batch in tqdm(pred_loader, desc="Predicting Categories"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            ids = batch["ID"]
            aspects = batch["Aspect"]

            embeddings = model(input_ids, mask).cpu().numpy()
            probs = knn.predict_proba(embeddings)

            THRESHOLD = 0.15

            for i, sent_id in enumerate(ids):
                row_probs = probs[i]

                active_indices = np.where(row_probs > THRESHOLD)[0]

                if len(active_indices) == 0:
                    active_indices = [np.argmax(row_probs)]

                pred_cats_list = label_encoder.inverse_transform(active_indices)
                final_cat = pred_cats_list[0]

                unique_key = f"{sent_id}_{aspects[i]}"
                results[unique_key] = final_cat

    print(f"Updating predictions in {target_file}...")

    final_output = []

    os.makedirs(os.path.dirname(target_file), exist_ok=True)

    with open(subtask2_pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            triplets = item.get("Triplet")
            new_quads = []

            for t in triplets:
                aspect = t.get("Aspect")
                opinion = t.get("Opinion")
                va = t.get("VA")

                lookup_key = f"{item['ID']}_{aspect}"
                predicted_cat = results.get(lookup_key, "NULL")

                new_quad = {
                    "Aspect": aspect,
                    "Category": predicted_cat,
                    "Opinion": opinion,
                    "VA": va
                }

                new_quads.append(new_quad)

            if "Triplet" in item:
                del item["Triplet"]
            item["Quadruplet"] = new_quads

            final_output.append(item)

    with open(target_file, 'w', encoding='utf-8') as f:
        for item in final_output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Predicted quadruplets saved to {target_file}")

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60

    print(f"\nTotal Execution Time: {elapsed_minutes:.2f} minutes ({elapsed_seconds:.0f} seconds)")


if __name__ == "__main__":
    main()
