import random
import sys
import os
import time
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModel, AutoTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared import config, utils_clean

try:
    import src.subtask_2.inference_clean as subtask2_inference
except ImportError:
    print("Error: Could not import src.subtask_2.inference. Make sure the file exists.")
    sys.exit(1)


class PKSampler(Sampler):
    """
    Ensures each batch contains at least K samples from P classes.
    """

    def __init__(self, data_source, p=16, k=8):
        self.data_source = data_source
        self.p = p
        self.k = k
        self.batch_size = p * k

        # Organize indices by label
        self.label_to_indices = defaultdict(list)
        for idx, item in enumerate(data_source.data):
            # We need the raw label string or ID here
            label = item["Category"]
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        # Estimate length based on total items
        num_batches = len(self.data_source) // self.batch_size
        return num_batches * self.batch_size

    def __iter__(self):
        # Create a pool of indices for each class
        label_pools = {l: list(idxs) for l, idxs in self.label_to_indices.items()}

        # Shuffle indices within pools
        for l in label_pools:
            random.shuffle(label_pools[l])

        # Generate batches
        # We perform roughly len(dataset) / batch_size iterations
        num_batches = len(self.data_source) // self.batch_size

        for _ in range(num_batches):
            batch_indices = []

            # 1. Randomly pick P classes
            selected_classes = random.sample(self.labels, self.p)

            for cls in selected_classes:
                # 2. Pick K instances from each class
                pool = label_pools[cls]

                if len(pool) >= self.k:
                    selected = [pool.pop() for _ in range(self.k)]
                else:
                    # If run out of data, refill with replacement (oversampling)
                    selected = random.choices(self.label_to_indices[cls], k=self.k)

                batch_indices.extend(selected)

            random.shuffle(batch_indices)  # Shuffle within batch to mix P and K
            yield from batch_indices

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
        self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(cls_embedding, dim=1)


def load_training_data(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"Error: Training file not found at {filepath}")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = item.get("Text")
            for q in item.get("Quadruplet", []):
                if q.get("Category"):
                    data.append({
                        "Text": text,
                        "Aspect": q.get("Aspect"),
                        "Category": q.get("Category"),
                        "Opinion": q.get("Opinion")
                    })
    return data


def load_inference_data(filepath):
    data = []
    if not os.path.exists(filepath):
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            for t in item.get("Triplet", []):
                data.append({
                    "ID": str(item.get("ID")),
                    "Text": item.get("Text"),
                    "Aspect": t.get("Aspect"),
                    "Opinion": t.get("Opinion")
                })
    return data



def main():
    start_time = time.time()

    model_name = config.TASK_CONFIGS["subtask_3"]["MODEL_NAME"]

    adapter_path = os.path.join(config.PROJECT_ROOT, "outputs", "subtask_2", "models", "final_adapter")

    subtask2_pred_file = os.path.join(config.PROJECT_ROOT, "outputs", "subtask_2", "predictions",
                                      f"pred_{config.LANG}_{config.DOMAIN}.jsonl")

    final_output_file = os.path.join(config.PROJECT_ROOT, "outputs", "subtask_3", "predictions",
                                     f"pred_{config.LANG}_{config.DOMAIN}.jsonl")

    print(f"\n{'=' * 30}\nSUBTASK 3: CONTRASTIVE CATEGORY EXTRACTION\n{'=' * 30}")
    print(f"Model: {model_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print("\n[Step 1/4] Running Subtask 2 Inference (Aspect, Opinion, VA)...")
    if os.path.exists(subtask2_pred_file):
        print(f"Found existing predictions at {subtask2_pred_file}. Skipping LLM Inference.")

    elif os.path.exists(adapter_path):
        subtask2_inference.run_inference(
            checkpoint_path=adapter_path,
            input_file=config.PREDICT_FILE,
            output_file=subtask2_pred_file,
            base_model_id="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
        )
    else:
        print(f"CRITICAL WARNING: Adapter not found at {adapter_path}")
        print("Cannot run Subtask 2. Assuming `pred_eng_laptop.jsonl` already exists in subtask_2/predictions.")
        if not os.path.exists(subtask2_pred_file):
            print("Error: Intermediate predictions missing. Aborting.")
            return

    print("\n[Step 2/4] Training Contrastive Encoder on Categories...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_training_data(config.TRAIN_FILE)

    all_categories = sorted(list(set(d["Category"] for d in train_data)))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_categories)
    print(f"Loaded {len(train_data)} training examples with {len(all_categories)} unique categories.")



    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = CategoryDataset(train_data, tokenizer, config.MAX_LEN, label_encoder, is_inference=False)

    sampler = PKSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=0
    )

    model = ContrastiveCategoryModel(model_name).to(device)
    model.backbone.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = utils_clean.SupConLoss(temperature=0.07)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            features1 = model(input_ids, mask)  # Returns (8, 1024)
            features2 = model(input_ids, mask)

            features = torch.cat([features1, features2], dim=0)
            doubled_labels = torch.cat([labels, labels], dim=0)

            #Directly pass the 2D tensor to the criterion


            loss = criterion(features, doubled_labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

    print("\n[Step 3/4] Building Class prototypes...")
    model.eval()

    memory_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    bank_embeddings = []
    bank_labels = []

    with torch.no_grad():
        for batch in tqdm(memory_loader, desc="Encoding Memory", leave=True):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            emb = model(input_ids, mask)
            bank_embeddings.append(emb.cpu().numpy())
            bank_labels.extend(batch["labels"].cpu().numpy())

    X_bank = np.vstack(bank_embeddings)
    y_bank = np.array(bank_labels)

    unique_classes = np.unique(y_bank)
    prototypes = {}

    for cls_id in unique_classes:
        vectors = X_bank[y_bank == cls_id]
        centroid = np.mean(vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        prototypes[cls_id] = centroid

    print(f"Computed prototypes for {len(prototypes)} classes.")

    print("\n[Step 4/4] Predicting Categories for Inference Data...")

    pred_data = load_inference_data(subtask2_pred_file)
    if not pred_data:
        print("No triplets found to classify.")
        return

    pred_dataset = CategoryDataset(pred_data, tokenizer, config.MAX_LEN, is_inference=True)
    pred_loader = DataLoader(pred_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    results_map = {}

    with torch.no_grad():
        for batch in tqdm(pred_loader, desc="Classifying"):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            ids = batch["ID"]
            aspects = batch["Aspect"]

            embeddings = model(input_ids, mask).cpu().numpy()

            proto_matrix = np.stack([prototypes[c] for c in sorted(prototypes.keys())])
            proto_labels = sorted(prototypes.keys())

            sims = np.dot(embeddings, proto_matrix.T)

            best_indices = np.argmax(sims, axis=1)

            for i, best_idx in enumerate(best_indices):
                pred_id = proto_labels[best_idx]
                final_cat = label_encoder.inverse_transform([pred_id])[0]

                unique_key = f"{ids[i]}_{aspects[i]}"
                results_map[unique_key] = final_cat

    final_output = []
    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)

    with open(subtask2_pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            triplets = item.get("Triplet", [])
            new_quads = []

            for t in triplets:
                aspect = t.get("Aspect")
                lookup_key = f"{item['ID']}_{aspect}"

                predicted_cat = results_map.get(lookup_key, "NULL")

                new_quad = {
                    "Aspect": aspect,
                    "Category": predicted_cat,
                    "Opinion": t.get("Opinion"),
                    "VA": t.get("VA")
                }
                new_quads.append(new_quad)

            if "Triplet" in item: del item["Triplet"]
            item["Quadruplet"] = new_quads
            final_output.append(item)

    with open(final_output_file, 'w', encoding='utf-8') as f:
        for item in final_output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Success! Final predictions saved to: {final_output_file}")

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nTotal Execution Time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()