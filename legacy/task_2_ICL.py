
"""
In-Context Learning + Regression pipeline for Subtask 2 (DimASTE)

Usage example:
python src/subtask_2/subtask2_icl_regression.py \
  --input_file "/mnt/data/eng_restaurant_train_alltasks (1).jsonl" \
  --output_file "data/subtask2/icl_output.jsonl" \
  --gen_model t5-base \
  --reg_model_checkpoint models/subtask1/best_regressor.pt \
  --reg_model_name bert-base-multilingual-cased \
  --icl_examples 6 \
  --device cuda \
  --max_gen_len 128
"""

import argparse
import json
import random
import re
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Project imports - adapt path if needed
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared.model import TransformerVARegressor   # assumes this exists


# -------------------------
# Helpers
# -------------------------
def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def save_jsonl(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_icl_prompt(examples: List[Dict], query_text: str, format_style: str = "compact") -> str:
    """
    Build an in-context prompt: several examples (Text -> Triplet), then the query.
    Each example is expected to have keys: 'Text' and 'Quadruplet' (list of dicts).
    Output format we ask the generator to produce is a compact JSON-like block:
      Triplet: [ (Aspect; Opinion), ... ]
    We do NOT request VA from the generator (regression step will handle VA).
    """
    parts = []
    for ex in examples:
        text = ex["Text"].strip()
        triplets = ex.get("Quadruplet", [])
        # Only include non-NULL triplets
        pairs = []
        for q in triplets:
            asp = q.get("Aspect", "").strip()
            opn = q.get("Opinion", "").strip()
            if not asp or asp.upper() == "NULL" or not opn or opn.upper() == "NULL":
                continue
            # keep original casing
            pairs.append(f'("{asp}", "{opn}")')
        pairs_str = ", ".join(pairs) if pairs else "[]"
        parts.append(f"Text: {text}\nTriplet: {pairs_str}\n")
    # Query
    parts.append(f"Text: {query_text.strip()}\nTriplet:")
    prompt = "\n".join(parts)
    return prompt


_triplet_pattern = re.compile(r'\("(.+?)"\s*,\s*"(.+?)"\)')

def parse_generated_triplets(gen_text: str) -> List[Tuple[str,str]]:
    """
    Parse generator output for triplet pairs like ("aspect", "opinion"), ...
    Returns list of (aspect, opinion)
    """
    pairs = []
    for m in _triplet_pattern.finditer(gen_text):
        a = m.group(1).strip()
        o = m.group(2).strip()
        if a and o:
            pairs.append((a, o))
    # Fallback: try to parse simple "Aspect: X; Opinion: Y" patterns
    if not pairs:
        # find lines with quotes "aspect" and "opinion"
        quoted = re.findall(r'"([^"]+)"', gen_text)
        if len(quoted) >= 2:
            # pair up sequentially
            it = iter(quoted)
            for a,b in zip(it, it):
                pairs.append((a.strip(), b.strip()))
    return pairs


def predict_va_for_phrase(reg_model: TransformerVARegressor, tokenizer, device, phrase: str, max_len: int = 128) -> Tuple[float,float]:
    """
    Predict (V,A) for a text phrase using regression model.
    We construct an input that the Subtask1 model expects — e.g., phrase alone or phrase+context.
    Here we feed the phrase alone.
    """
    enc = tokenizer(phrase, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    reg_model.to(device)
    reg_model.eval()
    with torch.no_grad():
        # assume reg_model has a forward that returns [batch, 2] or model.predict method
        out = reg_model.predict(input_ids=input_ids, attention_mask=attention_mask, device=device)
        # Accept either numpy/torch
        if isinstance(out, torch.Tensor):
            out = out.cpu().numpy()
        v, a = float(out[0][0]), float(out[0][1])
    return v, a


# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(
    input_file: str,
    output_file: str,
    gen_model_name: str,
    reg_model_checkpoint: str,
    reg_model_name: str,
    icl_examples: int,
    max_gen_len: int,
    temp: float,
    top_k: int,
    device: str,
    seed: int
):
    random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset (we expect labeled examples in this file so we can sample ICL examples)
    dataset = load_jsonl(input_file)

    # Prepare generator
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name, use_fast=True)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(device)

    # Prepare regression model: uses your Subtask1 TransformerVARegressor interface
    # We assume TransformerVARegressor has a method predict(...) that returns [batch,2] tensor/np
    reg_tokenizer = AutoTokenizer.from_pretrained(reg_model_name, use_fast=True)
    reg_model = TransformerVARegressor(model_name=reg_model_name)
    # Load checkpoint
    if not reg_model_checkpoint:
        raise ValueError("You must provide --reg_model_checkpoint path to the trained Subtask1 regressor.")
    ck = torch.load(reg_model_checkpoint, map_location="cpu", weights_only=False)
    # assume state dict is saved under "model_state" or directly
    if isinstance(ck, dict) and "model_state" in ck:
        reg_model.load_state_dict(ck["model_state"])
    else:
        reg_model.load_state_dict(ck)
    reg_model.to(device)
    reg_model.eval()

    outputs = []
    n = len(dataset)
    # Pre-select examples for ICL: sample from dataset those with non-NULL triplets
    candidate_examples = [d for d in dataset if any(q.get("Aspect","").upper()!="NULL" and q.get("Opinion","").upper()!="NULL" for q in d.get("Quadruplet", []))]
    # shuffle for randomness
    random.shuffle(candidate_examples)

    for idx, row in enumerate(dataset):
        text = row["Text"]
        id_ = row.get("ID", str(idx))
        # Build prompt using up to icl_examples from candidate_examples excluding current row if it has labels
        examples = []
        for ex in candidate_examples:
            if ex.get("ID") == id_:
                continue
            examples.append(ex)
            if len(examples) >= icl_examples:
                break

        prompt = build_icl_prompt(examples, text)

        # tokenize and generate
        input_ids = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
        gen_kwargs = dict(max_length=max_gen_len, temperature=temp, do_sample=True, top_k=top_k)
        gen_ids = gen_model.generate(input_ids, **gen_kwargs)
        gen_text = gen_tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # parse generated triplets (aspect, opinion) pairs
        pairs = parse_generated_triplets(gen_text)

        # If generator failed to produce any pairs, fallback to heuristic: look for quotes or split by commas for short opinion phrases
        if not pairs:
            # extremely simple heuristic: split on ',' and ' but '
            heuristics = []
            # attempt to find "but" conjunctions
            parts = re.split(r'\sbut\s|\band\b|;|,', text, flags=re.IGNORECASE)
            # naive mapping: assume first noun phrase is aspect + adjacent adj/opinion — this is fragile
            # We'll try to find quoted spans from dataset examples as fallback
            # For now, skip if can't parse
            pairs = []

        # For each pair, run regression on opinion phrase (or combined)
        triplet_out = []
        for asp, opn in pairs:
            # Choose phrase to pass to regressor: opinion phrase is a good proxy
            phrase_for_reg = opn
            try:
                v, a = predict_va_for_phrase(reg_model, reg_tokenizer, device, phrase_for_reg)
            except Exception:
                # fallback small default val
                v, a = 5.0, 5.0
            va_str = f"{v:.2f}#{a:.2f}"
            triplet_out.append({"Aspect": asp, "Opinion": opn, "VA": va_str})

        outputs.append({"ID": id_, "Triplet": triplet_out})

        # simple progress print
        if (idx + 1) % 50 == 0 or idx == n - 1:
            print(f"[{idx+1}/{n}] processed. Example output ID={id_}: {triplet_out[:3]}")

    # Save results
    save_jsonl(outputs, output_file)
    print("Saved outputs to:", output_file)


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Subtask2: In-Context Learning + Regression pipeline")
    p.add_argument("--input_file", type=str, default="/mnt/data/eng_restaurant_train_alltasks (1).jsonl",
                   help="Labeled JSONL (used to sample ICL examples) or unlabeled JSONL for inference.")
    p.add_argument("--output_file", type=str, default="data/subtask2/icl_output.jsonl")
    p.add_argument("--gen_model", type=str, default="t5-base", help="Seq2Seq model for ICL extraction")
    p.add_argument("--reg_model_checkpoint", type=str, required=True,
                   help="Checkpoint path for Subtask1 regression model (trained).")
    p.add_argument("--reg_model_name", type=str, default="bert-base-multilingual-cased",
                   help="Backbone name used by regression model (for tokenizer).")
    p.add_argument("--icl_examples", type=int, default=6, help="How many ICL examples to include in prompt")
    p.add_argument("--max_gen_len", type=int, default=128, help="Max tokens to generate")
    p.add_argument("--temp", type=float, default=0.7, help="Generation temperature")
    p.add_argument("--top_k", type=int, default=50, help="top_k sampling")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_file=args.input_file,
        output_file=args.output_file,
        gen_model_name=args.gen_model,
        reg_model_checkpoint=args.reg_model_checkpoint,
        reg_model_name=args.reg_model_name,
        icl_examples=args.icl_examples,
        max_gen_len=args.max_gen_len,
        temp=args.temp,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
    )

