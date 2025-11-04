import json
import requests
import re
import math
import pandas as pd
import numpy as np
import torch
from typing import List, Dict
from scipy.stats import pearsonr

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
        df = df.drop(columns=['VA', 'Opinion'])
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


def extract_num(s: str) -> int:
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1


def df_to_jsonl(df: pd.DataFrame, out_path: str):
    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
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

