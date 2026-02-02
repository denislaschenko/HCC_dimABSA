import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler
from src.shared import utils, config
from src.shared.dataset import VADataset
from src.shared.model import TransformerVARegressor
from scripts.vis.generate_results_plot import generate_plot

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ENSEMBLE_SEEDS = [100, 42, 2026]


def get_optimizer_params(model, base_lr, weight_decay=0.01, decay_factor=0.9):
    """
    Assigns lower learning rates to earlier layers of the transformer.
    """
    params = []
    # Get all transformer layers + embeddings
    layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    layers.reverse()  # Start from top (layer 23) to bottom (embeddings)

    lr = base_lr
    for layer in layers:
        params.append({"params": layer.parameters(), "lr": lr, "weight_decay": weight_decay})
        lr *= decay_factor  # Earlier layers get smaller LR

    # Final regression heads get full base_lr
    params.append({"params": model.v_head.parameters(), "lr": base_lr})
    params.append({"params": model.a_head.parameters(), "lr": base_lr})
    return params

def train_single_seed(seed, input_file):
    """
    Exakt die Logik deiner alten main(), aber parametrisiert mit seed.
    Gibt die Predictions zurück, statt sie direkt zu speichern.
    """
    utils.set_seed(seed)
    print(f"\n--- Starting run with SEED {seed} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerVARegressor(model_name=config.MODEL_NAME, num_bins=config.NUM_BINS).to(device)
    emb_weight = model.backbone.embeddings.word_embeddings.weight

    if torch.isnan(emb_weight).any():
        print("CRITICAL ERROR: Weights corrupted IMMEDIATELY after .to(device)!")
        return None
    else:
        print("Diagnostic: Model successfully secured on GPU.")

        opt_params = get_optimizer_params(model, config.LEARNING_RATE)
        optimizer = AdamW(opt_params, fused=False)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Daten laden (Wie im Original)
    train_raw = utils.load_jsonl(config.TRAIN_FILE)
    predict_raw = utils.load_jsonl(input_file)

    if not train_raw or not predict_raw:
        print("Error: Data loading failed.")
        return None

    train_df = utils.jsonl_to_df(train_raw)
    predict_df = utils.jsonl_to_df(predict_raw)

    # In src/subtask_1/train_subtask1_ens.py

    # --- ADD THIS SANITY CHECK ---
    print("\n--- DATA SANITY CHECK ---")
    if train_df.isnull().values.any():
        print("CRITICAL ERROR: NaN values found in Training Data!")
        print(train_df[train_df.isnull().any(axis=1)])
        return None  # Stop immediately

    # Check for labels out of bounds (assuming 1-9 range)
    # Adjust column names 'Valence'/'Arousal' if they differ in your DF
    if 'Valence' in train_df.columns:
        if train_df['Valence'].min() < 1.0 or train_df['Valence'].max() > 9.0:
            print("WARNING: Valence labels out of expected range [1, 9]")
            print(train_df['Valence'].describe())
    # -----------------------------

    # WICHTIG: Split muss stabil bleiben über Seeds hinweg für vergleichbare Metriken,
    # daher nutzen wir hier config.SEED statt dem Loop-Seed für den Split.
    train_df, dev_df = train_test_split(train_df, test_size=0.1, random_state=config.SEED)

    train_dataset = VADataset(train_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS, sigma=config.SIGMA)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0, drop_last=True)

    dev_dataset = VADataset(dev_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS, sigma=config.SIGMA)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=False, num_workers=0)

    predict_dataset = VADataset(predict_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS,
                                sigma=config.SIGMA)
    predict_loader = DataLoader(predict_dataset, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=False,
                                num_workers=0)

    loss_fn = utils.SafeLDLLoss()

    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(num_training_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Temporärer Pfad für diesen Seed, damit wir nicht überschreiben
    temp_save_path = config.MODEL_SAVE_PATH.replace(".pt", f"_seed_{seed}.pt")

    scaler = GradScaler()
    # Training Loop (unverändert)
    print("\n--- Starting Training ---")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")  # <--- Wieder eingefügt

        train_loss = model.train_epoch(train_loader, optimizer, scheduler, device, loss_fn, scaler)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")  # <--- Wieder eingefügt

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), temp_save_path)
            print(f"New best model saved for seed {seed} (Loss: {val_loss:.4f})")  # Optional: Bestätigung
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == config.PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Bestes Modell laden
    if os.path.exists(temp_save_path):
        print(f"Loading best model from {temp_save_path}...")
        model.load_state_dict(torch.load(temp_save_path))
    else:
        print(f"CRITICAL WARNING: No best model saved for seed {seed} (likely due to NaN loss).")
        print("Skipping evaluation for this seed.")
        return None

# TODO: check where the gold set came from
    pred_v_dev, pred_a_dev, gold_v_dev, gold_a_dev = utils.get_ldl_predictions(model, dev_loader, device, type="dev")
    pred_v_test, pred_a_test = utils.get_ldl_predictions(model, predict_loader, device, type="download")

    single_seed_score = utils.evaluate_predictions_task1(pred_a_dev, pred_v_dev, gold_a_dev, gold_v_dev)

    print(f"\n>>> RESULT FOR SEED {seed}:")
    print(f"    PCC_V:   {single_seed_score['PCC_V']:.4f}")
    print(f"    PCC_A:   {single_seed_score['PCC_A']:.4f}")
    print(f"    RMSE_VA: {single_seed_score['RMSE_VA']:.4f}")
    print(f"{'-' * 40}\n")

    del model
    torch.cuda.empty_cache()

    return {
        "dev_preds": (pred_v_dev, pred_a_dev),
        "dev_gold": (gold_v_dev, gold_a_dev),
        "test_preds": (pred_v_test, pred_a_test),
        "predict_df": predict_df
    }


def predict_va_for_subtask2(triplets_data: list, model_dir=None):
    """
    Callable function for Subtask 2 pipeline.
    Input: List of dicts (Subtask 2 format with 'ID', 'Text', 'Triplet').
           The 'Triplet' list contains dicts with 'Aspect' and 'Opinion'.
    Output: The same list, but with 'VA' fields populated (e.g., "7.24#7.14").
    """
    print(f"\n--- Running Subtask 1 VA Prediction on {len(triplets_data)} samples ---")

    # 1. Reuse Utils to flatten the complex Triplet structure into a DataFrame
    # This automatically handles multiple aspects/opinions per sentence
    predict_df = utils.jsonl_to_df(triplets_data)

    if predict_df.empty:
        print("Warning: Input data for VA prediction was empty.")
        return triplets_data

    REGRESSION_MODEL_NAME = "FacebookAI/roberta-large"
    print(f"Using Regression Model Architecture: {REGRESSION_MODEL_NAME}")

    # 2. Setup Data Loading for Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(REGRESSION_MODEL_NAME)

    # Create dataset (shuffle=False is crucial to keep order matching the dataframe)
    dataset = VADataset(predict_df, tokenizer, max_len=config.MAX_LEN, num_bins=config.NUM_BINS, sigma=config.SIGMA)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Load Ensemble Models & Predict
    all_pred_v = []
    all_pred_a = []

    models_found = 0

    if model_dir:
        # If called from subtask 2, construct path manually: {model_dir}/best_model_laptop_seed_{seed}.pt
        # We assume the filename convention is constant (best_model_{domain}_seed_{seed}.pt)
        base_filename = os.path.basename(config.MODEL_SAVE_PATH)  # e.g. "best_model_laptop.pt"
        base_path = os.path.join(model_dir, base_filename)
    else:
        # Default behavior (running inside subtask 1)
        base_path = config.MODEL_SAVE_PATH

    # Iterate over the seeds defined in your script
    for seed in ENSEMBLE_SEEDS:
        seed_path = base_path.replace(".pt", f"_seed_{seed}.pt")

        if not os.path.exists(seed_path):
            print(f"Warning: Model for seed {seed} not found at {seed_path}. Skipping.")
            continue

        print(f"Loading ensemble model: seed {seed}")
        model = TransformerVARegressor(model_name=REGRESSION_MODEL_NAME, num_bins=config.NUM_BINS).to(device)
        model.load_state_dict(torch.load(seed_path, map_location=device))
        model.eval()

        # Get predictions for this single model
        v, a = utils.get_ldl_predictions(model, dataloader, device, type="download")
        all_pred_v.append(v)
        all_pred_a.append(a)
        models_found += 1

        # Cleanup to save memory
        del model
        torch.cuda.empty_cache()

    if models_found == 0:
        raise FileNotFoundError("No trained models found! Run train_subtask1_ens.py to train first.")

    # 4. Average the predictions (Ensembling)
    avg_pred_v = np.mean(all_pred_v, axis=0)
    avg_pred_a = np.mean(all_pred_a, axis=0)

    # 5. Assign predictions back to the DataFrame
    predict_df["Valence"] = avg_pred_v
    predict_df["Arousal"] = avg_pred_a

    # 6. Reconstruct the Subtask 2 List Format
    # This logic mirrors utils.df_to_jsonl but returns a list instead of writing to a file

    # Helper to sort correctly by ID number
    predict_df["sort_key"] = predict_df["ID"].apply(utils.extract_num)
    df_sorted = predict_df.sort_values(by="sort_key")
    grouped = df_sorted.groupby("ID", sort=False)

    output_data = []

    for gid, gdf in grouped:
        triplets = []
        for _, row in gdf.iterrows():
            # Build the Triplet object
            t_obj = {
                "Aspect": row["Aspect"],
                "Opinion": row.get("Opinion", ""),  # Preserve Opinion if it exists
                "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
            }
            triplets.append(t_obj)

        # Reconstruct the sentence object
        # We take the text from the first row of the group
        text = gdf.iloc[0]["Text"]

        output_data.append({
            "ID": gid,
            "Text": text,
            "Triplet": triplets
        })

    return output_data

def main(input: str):
    print(f"Is CUDA available? {torch.cuda.is_available()}")

    all_dev_v = []
    all_dev_a = []
    all_test_v = []
    all_test_a = []

    gold_v = None
    gold_a = None
    final_predict_df = None

    # --- Schleife über Seeds ---
    for seed in ENSEMBLE_SEEDS:
        results = train_single_seed(seed, input)
        if results is None: continue

        # Sammeln
        all_dev_v.append(results["dev_preds"][0])
        all_dev_a.append(results["dev_preds"][1])
        all_test_v.append(results["test_preds"][0])
        all_test_a.append(results["test_preds"][1])

        # Einmalig statische Daten speichern
        if gold_v is None:
            gold_v = results["dev_gold"][0]
            gold_a = results["dev_gold"][1]
            final_predict_df = results["predict_df"]

    # --- Averaging ---
    print("\n--- Averaging Ensemble Predictions ---")
    avg_dev_v = np.mean(all_dev_v, axis=0)
    avg_dev_a = np.mean(all_dev_a, axis=0)

    avg_test_v = np.mean(all_test_v, axis=0)
    avg_test_a = np.mean(all_test_a, axis=0)

    print("\n--- Recalibrating Ensemble Predictions ---")
    avg_test_v = utils.recalibrate_predictions(avg_test_v, gold_v)
    avg_test_a = utils.recalibrate_predictions(avg_test_a, gold_a)

    final_predict_df["Valence"] = np.clip(avg_test_v, 1.0, 9.0)
    final_predict_df["Arousal"] = np.clip(avg_test_a, 1.0, 9.0)

    # --- Evaluation & Speichern (Exakt wie in deiner alten main) ---
    eval_score = utils.evaluate_predictions_task1(avg_dev_a.tolist(), avg_dev_v.tolist(), gold_a, gold_v)

    print(f"\n--- Dev Set Evaluation (Ensemble) ---")
    print(f"PCC_V: {eval_score['PCC_V']:.4f}")
    print(f"PCC_A: {eval_score['PCC_A']:.4f}")
    print(f"RMSE_VA: {eval_score['RMSE_VA']:.4f}")

    final_predict_df["Valence"] = avg_test_v
    final_predict_df["Arousal"] = avg_test_a

    os.makedirs(os.path.dirname(config.PREDICTION_FILE), exist_ok=True)

    utils.df_to_jsonl(final_predict_df, config.PREDICTION_FILE)
    print(f"Predictions saved to {config.PREDICTION_FILE}")
    # CSV Log
    results_data = {
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'experiment': f"{config.MODEL_VERSION_ID}_ENSEMBLE",
        'model': config.MODEL_NAME,
        'pcc_v': eval_score['PCC_V'],
        'pcc_a': eval_score['PCC_A'],
        'rmse_va': eval_score['RMSE_VA']
    }
    utils.log_results_to_csv(config.CSV_DIR, results_data)

    try:
        generate_plot()
    except Exception as e:
        print(f"Warning: Plot generation failed ({e}), but predictions were saved successfully.")
#fill
if __name__ == "__main__":
    main(input=config.PREDICT_FILE)