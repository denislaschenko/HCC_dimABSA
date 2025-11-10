#!/usr/bin/env python3
"""
train_subtask1.py — Train a transformer-based regressor for Dimensional ABSA (Subtask 1)

Usage example:
---------------
python train_subtask1.py \
    --train_file task-dataset/track_a/subtask_1/eng/train.jsonl \
    --predict_file task-dataset/track_a/subtask_1/eng/test.jsonl \
    --model_name bert-base-multilingual-cased \
    --epochs 10 --batch_size 8 --lr 2e-5 \
    --patience 2 --seed 42 --output_dir outputs/subtask1
"""

import os
import sys
import argparse
import logging
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from src.subtask_1.progress_visualization.plot_training_curves import plot_training_curves

# --- Project imports ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared import utils
from src.subtask_1.dataset import VADataset
from src.subtask_1.model import TransformerVARegressor
from src.subtask_1.progress_visualization.generate_results_plot import generate_plot


# ---------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Valence/Arousal regression model for dimABSA.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--predict_file", type=str, required=True, help="Path to prediction (test) JSONL file")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", help="HF model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training/evaluation")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length for tokenizer")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="outputs/subtask1", help="Directory for outputs and checkpoints")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for scheduler")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------
def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w", encoding="utf-8")],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    # Environment setup
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Model: {args.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load data
    train_raw = utils.load_jsonl(args.train_file)
    predict_raw = utils.load_jsonl(args.predict_file)
    if not train_raw or not predict_raw:
        raise RuntimeError("Data loading failed. Check the provided file paths.")

    train_df = utils.jsonl_to_df(train_raw)
    predict_df = utils.jsonl_to_df(predict_raw)
    train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=args.seed)

    logger.info(f"Loaded {len(train_df)} train, {len(dev_df)} dev, {len(predict_df)} predict samples.")

    # Datasets & loaders
    train_loader = DataLoader(VADataset(train_df, tokenizer, args.max_len),
                              batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(VADataset(dev_df, tokenizer, args.max_len),
                            batch_size=args.batch_size, shuffle=False)
    predict_loader = DataLoader(VADataset(predict_df, tokenizer, args.max_len),
                                batch_size=args.batch_size, shuffle=False)

    # Model, optimizer, scheduler
    model = TransformerVARegressor(model_name=args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    """
    logger.info("--- Starting Training ---")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = model.train_epoch(train_loader, optimizer, scheduler, device, loss_fn)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, checkpoint_path)
            logger.info(f"Model checkpoint saved: {checkpoint_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement ({epochs_no_improve}/{args.patience})")

        if epochs_no_improve >= args.patience:
            logger.info("Early stopping triggered.")
            break

    logger.info("--- Training Finished ---")
    """
    train_losses, val_losses = [], []
    logger.info("--- Starting Training ---")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss = model.train_epoch(train_loader, optimizer, scheduler, device, loss_fn)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, checkpoint_path)
            logger.info(f"Model checkpoint saved: {checkpoint_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            logger.info("Early stopping triggered.")
            break
    logger.info("--- Training Finished ---")
    # --- After training ---
    plot_training_curves(train_losses, val_losses, args.output_dir)


    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    logger.info("Loading best model for evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    pred_v, pred_a, gold_v, gold_a = utils.get_predictions(model, dev_loader, device, type="dev")
    eval_score = utils.evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)

    logger.info("\n--- Dev Set Evaluation ---")
    logger.info(f"PCC_V: {eval_score['PCC_V']:.4f}")
    logger.info(f"PCC_A: {eval_score['PCC_A']:.4f}")
    logger.info(f"RMSE_VA: {eval_score['RMSE_VA']:.4f}")
    logger.info("--------------------------")

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    logger.info("Running predictions on the test set...")
    pred_v, pred_a = utils.get_predictions(model, predict_loader, device, type="pred")
    predict_df["Valence"], predict_df["Arousal"] = pred_v, pred_a

    pred_output_file = os.path.join(args.output_dir, "predictions.jsonl")
    utils.df_to_jsonl(predict_df, pred_output_file)
    logger.info(f"Predictions saved: {pred_output_file}")

    # -----------------------------------------------------------------
    # Logging results
    # -----------------------------------------------------------------
    results_data = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "model": args.model_name,
        "seed": args.seed,
        "pcc_v": eval_score["PCC_V"],
        "pcc_a": eval_score["PCC_A"],
        "rmse_va": eval_score["RMSE_VA"],
    }

    results_file = os.path.join(args.output_dir, "results.csv")
    utils.log_results_to_csv(results_file, results_data)
    generate_plot()

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
