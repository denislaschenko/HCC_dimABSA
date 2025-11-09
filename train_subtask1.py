import sys
import os
import torch
import torch.nn as nn
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
# from src.subtask_1.progress_visualization.generate_results_plot import generate_results_plot

#


def main():
    utils.set_seed(42)

    print(f"Using model: {config.MODEL_NAME}")
    print(f"Loading tokenizer...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    print(f"Loading data from: {config.TRAIN_FILE}")
    train_raw = utils.load_jsonl(config.TRAIN_FILE)
    predict_raw = utils.load_jsonl(config.PREDICT_FILE)

    if not train_raw or not predict_raw:
        print("Error: Data loading failed. Check file paths in config.py")
        return

    train_df = utils.jsonl_to_df(train_raw)
    predict_df = utils.jsonl_to_df(predict_raw)

    train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=42)
    print(f"Data loaded: {len(train_df)} train, {len(dev_df)} dev, {len(predict_df)} predict samples.")

    train_dataset = VADataset(train_df, tokenizer, max_len=config.MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    dev_dataset = VADataset(dev_df, tokenizer, max_len=config.MAX_LEN)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    predict_dataset = VADataset(predict_df, tokenizer, max_len=config.MAX_LEN)
    predict_loader = DataLoader(predict_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = TransformerVARegressor(model_name=config.MODEL_NAME).to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.MSELoss()

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

    pred_v, pred_a, gold_v, gold_a = utils.get_predictions(model, dev_loader, device, type="dev")
    eval_score = utils.evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)

    print(f"\n--- Dev Set Evaluation ---")
    print(f"PCC_V: {eval_score['PCC_V']:.4f}")
    print(f"PCC_A: {eval_score['PCC_A']:.4f}")
    print(f"RMSE_VA: {eval_score['RMSE_VA']:.4f}")
    print("--------------------------")

    # --- Prediction ---
    print("Running predictions on the test set...")
    pred_v, pred_a = utils.get_predictions(model, predict_loader, device, type="pred")

    predict_df["Valence"] = pred_v
    predict_df["Arousal"] = pred_a

    os.makedirs(os.path.dirname(config.PREDICTION_FILE), exist_ok=True)

    generate_results_plot()

    utils.df_to_jsonl(predict_df, config.PREDICTION_FILE)
    print(f"Predictions saved to {config.PREDICTION_FILE}")


if __name__ == "__main__":
    main()

