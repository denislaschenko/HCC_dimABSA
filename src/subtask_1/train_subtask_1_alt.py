import sys
import os
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.subtask_1 import config
from src.shared import utils
from src.subtask_1.dataset import VADataset
from src.subtask_1.model import TransformerVARegressor
from scripts.vis.generate_results_plot import generate_plot


def main():
    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # 1. Load Data
    train_raw = utils.load_jsonl(config.TRAIN_FILE)
    predict_raw = utils.load_jsonl(config.PREDICT_FILE)  # Output from extract_aspects.py

    if not train_raw or not predict_raw:
        print("Error: Data loading failed.")
        return

    # 2. Process Prediction Data (Flattening Lists)
    # expected input: {"Aspect": ["a1", "a2"], "Opinion": ["o1", "o2"], ...}
    print("Flattening prediction data...")
    flat_predict_data = []

    for item in predict_raw:
        aspects = item.get("Aspect", [])
        opinions = item.get("Opinion", [])

        # Ensure aspects is a list (handle empty or malformed cases)
        if isinstance(aspects, list) and len(aspects) > 0:
            # Zip aspect and opinion if opinion exists, otherwise filler
            if not isinstance(opinions, list) or len(opinions) != len(aspects):
                opinions = [""] * len(aspects)

            for asp, op in zip(aspects, opinions):
                flat_predict_data.append({
                    "ID": item["ID"],
                    "Text": item["Text"],
                    "Aspect": asp,
                    "Opinion": op,
                    "VA": "0#0"  # Placeholder
                })
        else:
            # Handle cases with no aspects if necessary
            pass

    train_df = utils.jsonl_to_df(train_raw)
    predict_df = pd.DataFrame(flat_predict_data)

    # 3. Setup Training
    train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)

    train_dataset = VADataset(train_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS, sigma=config.SIGMA)
    dev_dataset = VADataset(dev_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS, sigma=config.SIGMA)
    predict_dataset = VADataset(predict_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS,
                                sigma=config.SIGMA)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    predict_loader = DataLoader(predict_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = TransformerVARegressor(model_name=config.MODEL_NAME, num_bins=config.NUM_BINS).to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = utils.LDLLoss()

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(len(train_loader) * config.EPOCHS * 0.1),
        num_training_steps=len(train_loader) * config.EPOCHS
    )

    # 4. Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("\n--- Starting Training ---")
    for epoch in range(config.EPOCHS):
        model.train_epoch(train_loader, optimizer, scheduler, device, loss_fn)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}: Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == config.PATIENCE: break

    # 5. Prediction
    print("Loading best model and predicting...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    pred_v, pred_a = utils.get_ldl_predictions(model, predict_loader, device, type="pred")

    # Add predictions to the flat dataframe
    predict_df["Valence"] = pred_v
    predict_df["Arousal"] = pred_a

    # 6. Regrouping (Flat -> Triplet Structure)
    # We group by ID to reconstruct the list of triplets for each document
    print("Regrouping results into Triplet format...")
    final_output = []

    # Iterate over unique IDs to maintain document structure
    for doc_id, group in predict_df.groupby("ID"):
        # Take Text from the first entry of the group
        text = group.iloc[0]["Text"]

        triplets = []
        for _, row in group.iterrows():
            triplets.append({
                "Aspect": row["Aspect"],
                "Opinion": row["Opinion"],
                # Format VA as "Valence#Arousal" string
                "VA": f"{row['Valence']:.4f}#{row['Arousal']:.4f}"
            })

        final_output.append({
            "ID": doc_id,
            "Text": text,
            "Triplet": triplets
        })

    # 7. Saving
    os.makedirs(os.path.dirname(config.PREDICTION_FILE), exist_ok=True)
    with open(config.PREDICTION_FILE, 'w', encoding='utf-8') as f:
        for entry in final_output:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {config.PREDICTION_FILE}")


if __name__ == "__main__":
    main()