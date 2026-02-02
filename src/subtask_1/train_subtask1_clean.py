import json
import sys
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.amp import GradScaler, autocast
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# TODO: switch clean paths
from src.shared import utils_clean, config
from src.shared.dataset_clean import VADataset
from src.shared.model_clean import TransformerVARegressor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn, scaler, ccc_loss_fn, bins):
    model.train()
    total_loss = 0
    valid_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        orig_scores = batch["orig_scores"].to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            v_logits, a_logits = model(input_ids, attention_mask)

            loss_ldl = loss_fn(v_logits, orig_scores[:, 0]) + (2.0 * loss_fn(a_logits, orig_scores[:, 1]))

            v_probs = torch.softmax(v_logits, dim=1)
            a_probs = torch.softmax(a_logits, dim=1)

            v_pred = torch.sum(v_probs * bins, dim=1)
            a_pred = torch.sum(a_probs * bins, dim=1)

            loss_ccc = ccc_loss_fn(v_pred, orig_scores[:, 0]) + (2.0 * ccc_loss_fn(a_pred, orig_scores[:, 1]))

            loss = loss_ldl + (0.5 * loss_ccc)

        if torch.isnan(loss):
            print(f"skipping batch {batch_idx} due to NaN loss")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        valid_batches += 1

    return total_loss / valid_batches if valid_batches > 0 else 0.0


def validate(model, dataloader, device, loss_fn, ccc_loss_fn, bins):
    model.eval()
    total_loss = 0
    valid_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            orig_scores = batch["orig_scores"].to(device)

            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                v_logits, a_logits = model(input_ids, attention_mask)

                loss_ldl = loss_fn(v_logits, orig_scores[:, 0]) + loss_fn(a_logits, orig_scores[:, 1])

                v_probs, a_probs = torch.softmax(v_logits, dim=1), torch.softmax(a_logits, dim=1)
                v_pred = torch.sum(v_probs * bins, dim=1)
                a_pred = torch.sum(a_probs * bins, dim=1)

                loss_ccc = ccc_loss_fn(v_pred, orig_scores[:, 0]) + ccc_loss_fn(a_pred, orig_scores[:, 1])

                loss = loss_ldl + (0.5 * loss_ccc)

            if not torch.isnan(loss):
                total_loss += loss.item()
                valid_batches += 1

    return total_loss / valid_batches if valid_batches > 0 else float('nan')


def get_optimizer_params(model, base_lr, weight_decay=0.01, decay_factor=0.9):
    """Layer-wise learning rate decay."""
    params = []
    layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    layers.reverse()

    lr = base_lr
    for layer in layers:
        params.append({"params": layer.parameters(), "lr": lr, "weight_decay": weight_decay})
        lr *= decay_factor

    params.append({"params": model.v_head.parameters(), "lr": base_lr})
    params.append({"params": model.a_head.parameters(), "lr": base_lr})
    return params


def train_single_seed(seed, input_file):
    utils_clean.set_seed(seed)
    print(f"\n{'=' * 20}\nRunning SEED {seed}\n{'=' * 20}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerVARegressor(
        model_name=config.MODEL_NAME,
        num_bins=config.NUM_BINS
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_raw = utils_clean.load_jsonl(config.TRAIN_FILE)
    predict_raw = utils_clean.load_jsonl(input_file)

    if not train_raw:
        print("CRITICAL ERROR: Training data not found.")
        return None

    train_df = utils_clean.jsonl_to_df(train_raw)
    predict_df = utils_clean.jsonl_to_df(predict_raw)

    if train_df.isnull().values.any():
        print("Warning: NaN values found in training data. Dropping...")
        train_df = train_df.dropna()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=config.SEED)
    train_idx, dev_idx = next(splitter.split(train_df, groups=train_df['Text']))

    train_df_split = train_df.iloc[train_idx]
    dev_df_split = train_df.iloc[dev_idx]

    train_df = train_df_split
    dev_df = dev_df_split

    print(f"Split sizes - Train: {len(train_df)}, Dev: {len(dev_df)}")

    train_dataset = VADataset(
        train_df,
        tokenizer,
        config.MAX_LEN,
        config.NUM_BINS,
        config.SIGMA,
        config.current_config.get("INCLUDE_OPINION")
    )
    dev_dataset = VADataset(
        dev_df,
        tokenizer,
        config.MAX_LEN,
        config.NUM_BINS,
        config.SIGMA,
        config.current_config.get("INCLUDE_OPINION")
    )
    predict_dataset = VADataset(
        predict_df,
        tokenizer,
        config.MAX_LEN,
        config.NUM_BINS,
        config.SIGMA,
        config.current_config.get("INCLUDE_OPINION")
    )
# TODO: check pin_mem
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.NUM_WORKERS > 0 else False
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers = config.NUM_WORKERS,
        pin_memory = True if config.NUM_WORKERS > 0 else False
    )
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers = config.NUM_WORKERS,
        pin_memory = True if config.NUM_WORKERS > 0 else False
    )

    opt_params = get_optimizer_params(model, config.LEARNING_RATE)
    optimizer = AdamW(opt_params)

    num_training_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_training_steps * 0.1), num_training_steps)
    scaler = GradScaler()

    loss_fn = utils_clean.LDLLoss(sigma=config.SIGMA, num_bins=config.NUM_BINS).to(device)
    ccc_loss_fn = utils_clean.CCCLoss().to(device)
    bins = torch.linspace(1.0, 9.0, config.NUM_BINS).to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    temp_save_path = config.MODEL_SAVE_PATH.replace(".pt", f"_seed_{seed}.pt")

    os.makedirs(os.path.dirname(temp_save_path), exist_ok=True)

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn, scaler, ccc_loss_fn, bins)
        val_loss = validate(model, dev_loader, device, loss_fn, ccc_loss_fn, bins)

        print(f"Epoch {epoch + 1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), temp_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    if os.path.exists(temp_save_path):
        print(f"Loading best model for inference (Loss: {best_val_loss:.4f})...")
        model.load_state_dict(torch.load(temp_save_path))
    else:
        print("Warning: No model saved (NaN loss?). Skipping seed.")
        return None

    pred_v_dev, pred_a_dev, gold_v_dev, gold_a_dev = utils_clean.get_ldl_predictions(
        model,
        dev_loader,
        device,
        type="dev",
        num_bins=config.NUM_BINS
    )
    pred_v_test, pred_a_test = utils_clean.get_ldl_predictions(
        model,
        predict_loader,
        device,
        type="download",
        num_bins=config.NUM_BINS
    )

    metrics = utils_clean.evaluate_predictions_task1(pred_a_dev, pred_v_dev, gold_a_dev, gold_v_dev)
    print(f"Seed {seed} Results: PCC_V: {metrics['PCC_V']:.4f}, PCC_A: {metrics['PCC_A']:.4f}, RMSE: {metrics['RMSE_VA']:.4f}")

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return {
        "dev_preds": (pred_v_dev, pred_a_dev),
        "dev_gold": (gold_v_dev, gold_a_dev),
        "test_preds": (pred_v_test, pred_a_test),
        "predict_df": predict_df
    }


def predict_va_for_subtask2(triplets_data: list, model_dir=None):
    """
    Interface for Subtask 2 to call this model on extracted triplets.
    """
    print(f"\n--- Subtask 1 VA Inference on {len(triplets_data)} samples ---")

    predict_df = utils_clean.jsonl_to_df(triplets_data)
    if predict_df.empty:
        return triplets_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    dataset = VADataset(
        predict_df,
        tokenizer,
        config.MAX_LEN,
        config.NUM_BINS,
        config.SIGMA,
        include_opinion=config.current_config.get("INCLUDE_OPINION")
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.NUM_WORKERS > 0 else False
    )

    all_pred_v, all_pred_a = [], []
    base_save_path = config.MODEL_SAVE_PATH
    if model_dir:
        filename = os.path.basename(config.MODEL_SAVE_PATH)
        base_save_path = os.path.join(model_dir, filename)

    models_found = 0
    for seed in config.ENSEMBLE_SEEDS:
        seed_path = base_save_path.replace(".pt", f"_seed_{seed}.pt")
        if not os.path.exists(seed_path):
            continue

        model = TransformerVARegressor(model_name=config.MODEL_NAME, num_bins=config.NUM_BINS).to(device)
        model.load_state_dict(torch.load(seed_path, map_location=device))

        v, a = utils_clean.get_ldl_predictions(model, dataloader, device, type="download", num_bins=config.NUM_BINS)
        all_pred_v.append(v)
        all_pred_a.append(a)
        models_found += 1
        del model

    if models_found == 0:
        print("Warning: No models found. Returning original data.")
        return triplets_data

    avg_v = np.mean(all_pred_v, axis=0)
    avg_a = np.mean(all_pred_a, axis=0)

    predict_df["Valence"] = avg_v
    predict_df["Arousal"] = avg_a

    predict_df["sort_key"] = predict_df["ID"].apply(utils_clean.extract_num)
    df_sorted = predict_df.sort_values(by="sort_key")
    grouped = df_sorted.groupby("ID", sort=False)

    output_data = []
    for gid, gdf in grouped:
        triplets = []
        for _, row in gdf.iterrows():
            triplets.append({
                "Aspect": row["Aspect"],
                "Opinion": row.get("Opinion", ""),
                "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
            })
        output_data.append({
            "ID": gid,
            "Text": gdf.iloc[0]["Text"],
            "Triplet": triplets
        })

    return output_data


def main(input_path):
    start_time = time.time()

    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Config: {config.MODEL_NAME}, Epochs: {config.EPOCHS}, Bins: {config.NUM_BINS}")

    results_collection = []

    for seed in config.ENSEMBLE_SEEDS:
        res = train_single_seed(seed, input_path)
        if res:
            results_collection.append(res)

    if not results_collection:
        print("No successful runs.")
        return

    print("\n--- Aggregating Ensemble ---")
    all_dev_v = np.vstack([r["dev_preds"][0] for r in results_collection])
    all_dev_a = np.vstack([r["dev_preds"][1] for r in results_collection])
    all_test_v = np.vstack([r["test_preds"][0] for r in results_collection])
    all_test_a = np.vstack([r["test_preds"][1] for r in results_collection])

    avg_dev_v = np.mean(all_dev_v, axis=0)
    avg_dev_a = np.mean(all_dev_a, axis=0)
    avg_test_v = np.mean(all_test_v, axis=0)
    avg_test_a = np.mean(all_test_a, axis=0)

    gold_v = results_collection[0]["dev_gold"][0]
    gold_a = results_collection[0]["dev_gold"][1]

    score = utils_clean.evaluate_predictions_task1(avg_dev_a.tolist(), avg_dev_v.tolist(), gold_a, gold_v)
    print(
        f"\nENSEMBLE DEV RESULTS:\nPCC_V: {score['PCC_V']:.4f}\nPCC_A: {score['PCC_A']:.4f}\nRMSE: {score['RMSE_VA']:.4f}")

    final_df = results_collection[0]["predict_df"].copy()
    final_df["Valence"] = np.clip(avg_test_v, 1.0, 9.0)
    final_df["Arousal"] = np.clip(avg_test_a, 1.0, 9.0)

    os.makedirs(os.path.dirname(config.OUTPUT_FILE), exist_ok=True)
    utils_clean.df_to_jsonl(final_df, config.OUTPUT_FILE)

    utils_clean.log_results_to_csv(config.CSV_FILE, {
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'experiment': 'ENSEMBLE_CLEAN',
        'model': config.MODEL_NAME,
        'pcc_v': score['PCC_V'],
        'pcc_a': score['PCC_A'],
        'rmse_va': score['RMSE_VA']
    })
    print(f"Predictions saved to {config.OUTPUT_FILE}")

    end_time = time.time()

    time_passed = (end_time - start_time) / 60
    print(f"\nTotal Execution Time: {time_passed:.2f} minutes")

if __name__ == "__main__":
    main(input_path=config.PREDICT_FILE)