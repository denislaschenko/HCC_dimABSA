import csv
import json
import random
import re
import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional
from scipy.stats import pearsonr
from typing import List, Dict, Tuple



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss."""

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        batch_size = features.shape[0]

        features = functional.normalize(features, dim=1)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if mask is None:
            if labels.dim() == 1:
                labels = labels.contiguous().view(-1, 1)
                mask = torch.eq(labels, labels.T).float().to(device)
            else:
                mask = torch.eq(labels, labels).all(dim=1).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        loss = - self.temperature * mean_log_prob_pos
        loss = loss.mean()

        return loss


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient Loss."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        cov_matrix = torch.cov(torch.stack([x, y]))
        covariance = cov_matrix[0, 1]
        var_x = cov_matrix[0, 0]
        var_y = cov_matrix[1, 1]
        mean_diff = x.mean() - y.mean()
        ccc = (2 * covariance) / (var_x + var_y + mean_diff ** 2 + 1e-8)
        return 1.0 - ccc


class LDLLoss(nn.Module):
    """
    Label Distribution Learning Loss (KL Divergence).
    Dynamically generates the target distribution based on the scalar target.
    """
    def __init__(self, sigma=1.0, num_bins=9, min_val=1.0, max_val=9.0):
        super().__init__()
        self.sigma = sigma
        self.bins = torch.linspace(min_val, max_val, num_bins)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, target_scalar):
        device = logits.device

        if self.bins.device != device:
            self.bins = self.bins.to(device)

        target_scalar = target_scalar.float().view(-1, 1)
        logits = logits.float()

        bin_diff = self.bins.unsqueeze(0) - target_scalar
        target_dist = torch.exp(-(bin_diff ** 2) / (2 * self.sigma ** 2))

        target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-9)

        log_probs = functional.log_softmax(logits, dim=1)
        return self.kl_loss(log_probs, target_dist)

def load_jsonl(filepath: str) -> List[Dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON in file: {filepath}")
        return []


def jsonl_to_df(data: List[Dict]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
    elif 'Aspect' in data[0]:
        df = pd.DataFrame(data)
        if 'Aspect' in df.columns and df['Aspect'].apply(lambda x: isinstance(x, list)).any():
            df = df.explode('Aspect')
    else:
        df = pd.DataFrame(data)

    if 'VA' in df.columns:
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)

    df = df.drop(columns=['VA', 'Category'], errors='ignore')

    subset_cols = ['ID', 'Aspect']
    if 'Opinion' in df.columns:
        subset_cols.append('Opinion')

    df = df.dropna(subset=subset_cols)
    df = df.drop_duplicates(subset=subset_cols, keep='first')

    return df


def df_to_jsonl(df: pd.DataFrame, out_path: str):
    """Writes a DataFrame back to the competition JSONL format."""
    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)

    has_opinion = "Opinion" in df.columns

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            if has_opinion:
                record = {
                    "ID": gid,
                    "Triplet": []
                }
                for _, row in gdf.iterrows():
                    record["Triplet"].append({
                        "Aspect": row["Aspect"],
                        "Opinion": row["Opinion"],
                        "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                    })
            else:
                record = {
                    "ID": gid,
                    "Aspect_VA": []
                }
                for _, row in gdf.iterrows():
                    record["Aspect_VA"].append({
                        "Aspect": row["Aspect"],
                        "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                    })
            f.write(json.dumps(record, ensure_ascii=False) + "\n")



def get_ldl_predictions(model, dataloader, device, type="dev", num_bins=9):
    """
    Runs inference using Label Distribution Learning (LDL).
    Returns raw scalar predictions by calculating the expected value of the distribution.
    """
    model.eval()
    pred_v = []
    pred_a = []
    gold_v = []
    gold_a = []

    bins = torch.linspace(1.0, 9.0, num_bins).to(device)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            v_logits, a_logits = model(input_ids, attention_mask)

            v_probs = torch.softmax(v_logits, dim=1)
            a_probs = torch.softmax(a_logits, dim=1)

            v_scores = torch.sum(v_probs * bins, dim=1)
            a_scores = torch.sum(a_probs * bins, dim=1)

            pred_v.extend(v_scores.cpu().tolist())
            pred_a.extend(a_scores.cpu().tolist())

            if type == "dev" and "orig_scores" in batch:
                orig = batch["orig_scores"]  # shape [batch, 2]
                gold_v.extend(orig[:, 0].tolist())
                gold_a.extend(orig[:, 1].tolist())

    if type == "dev":
        return pred_v, pred_a, gold_v, gold_a
    else:
        return pred_v, pred_a


def evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v) -> dict:
    pcc_v = pearsonr(pred_v, gold_v)[0]
    pcc_a = pearsonr(pred_a, gold_a)[0]

    gold_va = np.array(gold_v) + np.array(gold_a)
    pred_va = np.array(pred_v) + np.array(pred_a)

    rmse_va = np.sqrt(np.mean((gold_va - pred_va) ** 2))

    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
    }


def recalibrate_predictions(preds, gold):
    """Matches the mean and std of predictions to the gold set (domain adaptation trick)."""
    preds = np.array(preds)
    gold = np.array(gold)

    mu_pred, std_pred = np.mean(preds), np.std(preds)
    mu_gold, std_gold = np.mean(gold), np.std(gold)

    if std_pred == 0:
        return preds

    recalibrated = (preds - mu_pred) * (std_gold / std_pred) + mu_gold
    return recalibrated



def extract_num(s: str) -> int:
    """Extracts the number from an ID string (e.g., '123' or 'id_123')."""
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def log_results_to_csv(csv_path: str, results: dict):
    try:
        row_data = [
            results.get('date', ''),
            results.get('experiment', ''),
            results.get('model', ''),
            f"{results.get('pcc_v', 0):.4f}",
            f"{results.get('pcc_a', 0):.4f}",
            f"{results.get('rmse_va', 0):.4f}"
        ]

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        print(f"Results appended to {csv_path}.")
    except Exception as e:
        print(f"Error logging to CSV: {e}")