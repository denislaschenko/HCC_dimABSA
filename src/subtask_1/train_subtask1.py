import sys
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared import utils, config
from src.shared.dataset import VADataset
from src.shared.model import TransformerVARegressor
from scripts.vis.generate_results_plot import generate_plot


def main(input: str):
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
    print(f"Torch Version: {torch.__version__}")

    version = 3
    utils.set_seed(config.SEED)

    print(f"Using model: {config.MODEL_NAME}")
    print(f"Loading tokenizer...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    print(f"Loading data from: {config.TRAIN_FILE}")
    train_raw = utils.load_jsonl(config.TRAIN_FILE)
    predict_raw = utils.load_jsonl(input)

    if not train_raw or not predict_raw:
        print("Error: Data loading failed. Check file paths in config.py")
        return

    train_df = utils.jsonl_to_df(train_raw)
    predict_df = utils.jsonl_to_df(predict_raw)

    # # print("DEBUG: Slicing training data to 50 samples for speed test.")
    # train_df = train_df.iloc[:50]

    train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=config.SEED)
    print(f"Data loaded: {len(train_df)} train, {len(dev_df)} dev, {len(predict_df)} predict samples.")

    train_dataset = VADataset(train_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS, sigma=config.SIGMA)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)

    dev_dataset = VADataset(dev_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS, sigma=config.SIGMA)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    predict_dataset = VADataset(predict_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS, sigma=config.SIGMA)
    predict_loader = DataLoader(predict_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)

    model = TransformerVARegressor(model_name=config.MODEL_NAME, num_bins=config.NUM_BINS).to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = utils.LDLLoss()

    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    print("\n--- Starting Training ---")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

        train_loss = model.train_epoch(train_loader, optimizer, scheduler, device, loss_fn)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Model saved to {config.MODEL_SAVE_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == config.PATIENCE:
            print(f"Early stopping at epoch {epoch + 1} as val_loss did not improve for {config.PATIENCE} epochs.")
            break

    print("--- Training Finished ---")


    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))

    pred_v, pred_a, gold_v, gold_a = utils.get_ldl_predictions(model, dev_loader, device, type="dev")
    eval_score = utils.evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)

    print(f"\n--- Dev Set Evaluation ---")
    print(f"PCC_V: {eval_score['PCC_V']:.4f}")
    print(f"PCC_A: {eval_score['PCC_A']:.4f}")
    print(f"RMSE_VA: {eval_score['RMSE_VA']:.4f}")
    print("--------------------------")

    print("Running predictions on the test set...")
    pred_v, pred_a = utils.get_ldl_predictions(model, predict_loader, device, type="pred")

    predict_df["Valence"] = pred_v
    predict_df["Arousal"] = pred_a

    os.makedirs(os.path.dirname(config.PREDICTION_FILE), exist_ok=True)

    print("\n--- Protokollierung der Ergebnisse ---")
    results_data = {
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'experiment': config.MODEL_VERSION_ID,
        'model': config.MODEL_NAME,
        'pcc_v': eval_score['PCC_V'],
        'pcc_a': eval_score['PCC_A'],
        'rmse_va': eval_score['RMSE_VA']
    }

    utils.log_results_to_csv(config.CSV_DIR, results_data)
    generate_plot()

    utils.df_to_jsonl(predict_df, config.PREDICTION_FILE)
    print(f"Predictions saved to {config.PREDICTION_FILE}")


if __name__ == "__main__":
    main(input=config.PREDICT_FILE)

