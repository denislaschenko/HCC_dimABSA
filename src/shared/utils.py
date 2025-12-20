import csv
import json
import random
import re
import math
import pandas as pd
import numpy as np
import torch
from typing import List, Dict
from scipy.stats import pearsonr
from torch import nn
from torch.nn import functional

from src.shared import config


class LDLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, targets):
        log_probs = functional.log_softmax(logits, dim=1)
        return self.kldiv(log_probs, targets)

class CCCLoss(nn.Module):
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
    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Category', 'Opinion'])
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')
    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA'])
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')
    elif 'Aspect_VA' in data[0]:
        df = pd.json_normalize(data, 'Aspect_VA', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')
    elif 'Aspect' in data[0]:
        df = pd.json_normalize(data, 'Aspect', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})
        df['Valence'] = 0  # default value
        df['Arousal'] = 0  # default value
    else:
        raise ValueError("Invalid format: must include 'Quadruplet', 'Triplet', or 'Aspect'")
    return df


def get_predictions(model, dataloader, device, type="dev") -> tuple:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask).cpu().numpy()
            all_preds.append(outputs)

            if type == "dev":
                labels = batch["labels"].cpu().numpy()
                all_labels.append(labels)

    preds = np.vstack(all_preds)
    pred_v = preds[:, 0]
    pred_a = preds[:, 1]

    if type == "dev":
        labels = np.vstack(all_labels)
        gold_v = labels[:, 0]
        gold_a = labels[:, 1]
        return pred_v, pred_a, gold_v, gold_a
    else:
        return pred_v, pred_a


def evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v) -> dict:
    if not (all(1 <= x <= 9 for x in pred_v) and all(1 <= x <= 9 for x in pred_a)):
        print(f"Warning: Some predicted values are out of the 1-9 numerical range.")

    pcc_v = pearsonr(pred_v, gold_v)[0]
    pcc_a = pearsonr(pred_a, gold_a)[0]

    gold_va = gold_v + gold_a
    pred_va = pred_v + pred_a

    def rmse(gold, pred):
        result = [(a - b) ** 2 for a, b in zip(gold, pred)]
        return math.sqrt(sum(result) / len(gold))

    rmse_va = rmse(gold_va, pred_va)

    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
    }

def get_ldl_predictions(model, dataloader, device, type="dev", num_bins=config.NUM_BINS):
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

            if type == "dev":
                if "orig_scores" in batch:
                    orig = batch["orig_scores"]  # shape [batch, 2]
                    gold_v.extend(orig[:, 0].tolist())
                    gold_a.extend(orig[:, 1].tolist())
                else:
                    pass
    if type == "dev":
        return pred_v, pred_a, gold_v, gold_a
    else:
        return pred_v, pred_a

def extract_num(s: str) -> int:
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1


def df_to_jsonl(df: pd.DataFrame, out_path: str):
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
            else :
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def log_results_to_csv(csv_path: str, results: dict):
    try:
        row_data = [
            results['date'],
            results['experiment'],
            results['model'],
            f"{results['pcc_v']:.4f}",
            f"{results['pcc_a']:.4f}",
            f"{results['rmse_va']:.4f}"
        ]

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)


            writer.writerow(row_data)

        print(f"successfully appended results to {csv_path}.")

    except IOError as e:
        print(f"error appending results to {csv_path}: {e}")

    except KeyError as e:
        print(f"ERROR: the key {e} could not be found, nothing has been appended.")