import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
import copy

import config
import utils
from src.subtask_1.dataset import VADataset
from src.subtask_1.model import TransformerVARegressor


def run():
    print(f"Loading configuration for {config.LANG}_{config.DOMAIN}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer for: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    print(f"Loading training data from: {config.TRAIN_URL}")
    train_raw = utils.load_jsonl_url(config.TRAIN_URL)
    train_full_df = utils.jsonl_to_df(train_raw)

    train_df, dev_df = train_test_split(train_full_df, test_size=0.1, random_state=42)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(dev_df)}")

    train_dataset = VADataset(train_df, tokenizer, config.MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    dev_dataset = VADataset(dev_df, tokenizer, config.MAX_LEN)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Initializing model: {config.MODEL_NAME}")
    model = TransformerVARegressor(config.MODEL_NAME).to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.MSELoss()

    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * 0.1)  # 10% warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    best_model_state = None

    print("Starting training...")
    for epoch in range(config.EPOCHS):
        train_loss = model.train_epoch(train_loader, optimizer, loss_fn, device, scheduler)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best model saved with Val Loss: {best_val_loss:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    print("Evaluating best model on dev set...")
    pred_v, pred_a, gold_v, gold_a = utils.get_predictions(model, dev_loader, device, type="dev")
    eval_score = utils.evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)

    print(f"\n--- {config.MODEL_NAME} Dev Evaluation ---")
    print(f"  PCC_V: {eval_score['PCC_V']:.4f}")
    print(f"  PCC_A: {eval_score['PCC_A']:.4f}")
    print(f"  RMSE_VA: {eval_score['RMSE_VA']:.4f}")
    print("--------------------------------------\n")

    print(f"Loading prediction data from: {config.PREDICT_URL}")
    predict_raw = utils.load_jsonl_url(config.PREDICT_URL)
    predict_df = utils.jsonl_to_df(predict_raw)

    predict_dataset = VADataset(predict_df, tokenizer, config.MAX_LEN)
    predict_loader = DataLoader(predict_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print("Generating final predictions...")
    pred_v, pred_a = utils.get_predictions(model, predict_loader, device, type="pred")

    predict_df["Valence"] = pred_v
    predict_df["Arousal"] = pred_a

    utils.df_to_jsonl(predict_df, config.OUTPUT_FILE)
    print(f"Submission file saved to: {config.OUTPUT_FILE}")


if __name__ == "__main__":
    run()