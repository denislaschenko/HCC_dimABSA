import sys
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional
import time

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.subtask_1 import config
from src.shared import utils
from src.subtask_1.dataset import VADataset
from src.subtask_1.model import TransformerVARegressor
from scripts.vis import generate_results_plot


def main(override_config: Optional[Dict[str, Any]] = None) -> float:
    trial_start_time = time.time()
    if override_config is None:
        override_config = {}

    model_name = override_config.get("model_name", config.MODEL_NAME)
    learning_rate = override_config.get("lr", config.LEARNING_RATE)
    batch_size = override_config.get("batch_size", config.BATCH_SIZE)
    epochs = override_config.get("epochs", config.EPOCHS)
    patience = override_config.get("patience", config.PATIENCE)
    model_version_id = override_config.get("version_id", config.MODEL_VERSION_ID)
    dropout_rate = override_config.get("dropout", 0.1)

    utils.set_seed(25)

    print(f"\n--- STARTING TRIAL: {model_version_id} ---")
    print(f"Model: {model_name}, LR: {learning_rate}, BS: {batch_size}, Dropout: {dropout_rate}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading data from: {config.TRAIN_FILE}")
    train_raw = utils.load_jsonl(config.TRAIN_FILE)
    predict_raw = utils.load_jsonl(config.PREDICT_FILE)

    if not train_raw or not predict_raw:
        print("Error: Data loading failed. Check file paths in config.py")
        return 999.0

    train_df = utils.jsonl_to_df(train_raw)
    predict_df = utils.jsonl_to_df(predict_raw)

    train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=25)
    print(f"Data loaded: {len(train_df)} train, {len(dev_df)} dev, {len(predict_df)} predict samples.")

    train_dataset = VADataset(train_df, tokenizer, max_len=config.MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = VADataset(dev_df, tokenizer, max_len=config.MAX_LEN)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    predict_dataset = VADataset(predict_df, tokenizer, max_len=config.MAX_LEN)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerVARegressor(model_name=model_name, dropout=dropout_rate).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = utils.CCCLoss()

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = model.train_epoch(train_loader, optimizer, scheduler, device, loss_fn)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch + 1} as val_loss did not improve for {patience} epochs.")
            break

    print("--- Training Finished ---")

    print("Loading best model for evaluation...")
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    else:
        print("Warning: No best model saved. Using model from final epoch.")

    pred_v, pred_a, gold_v, gold_a = utils.get_predictions(model, dev_loader, device, type="dev")
    eval_score = utils.evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)

    print(f"\n--- Dev Set Evaluation ---")
    print(f"PCC_V: {eval_score['PCC_V']:.4f}")
    print(f"PCC_A: {eval_score['PCC_A']:.4f}")
    print(f"RMSE_VA: {eval_score['RMSE_VA']:.4f}")
    print("--------------------------")

    print("Running predictions on the test set...")
    pred_v, pred_a = utils.get_predictions(model, predict_loader, device, type="pred")

    predict_df["Valence"] = pred_v
    predict_df["Arousal"] = pred_a

    os.makedirs(os.path.dirname(config.PREDICTION_FILE), exist_ok=True)
    utils.df_to_jsonl(predict_df, config.PREDICTION_FILE)
    print(f"Predictions saved to {config.PREDICTION_FILE}")

    print("\n--- Protokollierung der Ergebnisse ---")
    results_data = {
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'experiment': model_version_id,
        'model': model_name,
        'pcc_v': eval_score['PCC_V'],
        'pcc_a': eval_score['PCC_A'],
        'rmse_va': eval_score['RMSE_VA']
    }

    utils.log_results_to_csv("../../outputs/subtask_1/logs/results.csv", results_data)

    trial_end_time = time.time()
    total_seconds = trial_end_time - trial_start_time
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    print(f"--- TRIAL TIME: {minutes:02d}m {seconds:02d}s ---")

    return eval_score['RMSE_VA']


if __name__ == "__main__":

    final_rmse = main()

    print("\n--- MANUELLER LAUF ABGESCHLOSSEN ---")
    print(f"Finaler RMSE (an Optuna zurückgegeben): {final_rmse:.4f}")

    print("Running plotting script...")
    try:
        generate_results_plot.generate_plot()
        print("Plotting complete.")
    except Exception as e:
        print(f"Error running plot script: {e}")
