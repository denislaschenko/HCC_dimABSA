"""
python src/subtask_2/task2_icl_regression.py \
  --input_file /data/eng_restaurant_train_alltasks.jsonl \
  --output_file data/subtask2/preds_task2.jsonl \
  --reg_checkpoint subtask1/best_model_laptop_42.pt \
  --gen_model google/flan-t5-large \
  --reg_model_name bert-base-multilingual-cased \
  --device cuda \
  --icl_examples 10 \
  --score_opinion
"""
import os
import json
import argparse
import random
import torch
from typing import List, Dict

from src.subtask_2.regression_loader import build_regressor_from_checkpoint
from src.subtask_2.icl_triplet_extractor import ICLGenerator
from transformers import AutoTokenizer

# Import your regressor class (must exist)
from src.shared.model import TransformerVARegressor


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def save_jsonl(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def predict_va_with_regressor(regressor, tokenizer, device, phrase: str, max_len: int = 128):
    """
    Use regressor to predict V and A for a phrase.
    The function tries several possible regressor APIs:
      - regressor.predict(input_ids=..., attention_mask=..., device=...)
      - regressor.forward(input_ids=..., attention_mask=...) -> outputs
      - regressor(input_ids=..., attention_mask=...)
    Adjust to your regressor's actual interface if needed.
    """
    enc = tokenizer(phrase, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device)

    regressor.to(device)
    regressor.eval()

    with torch.no_grad():
        # try typical custom predict method
        if hasattr(regressor, "predict"):
            out = regressor.predict(input_ids=input_ids, attention_mask=att, device=device)
            # ensure tensor -> numpy
            if isinstance(out, torch.Tensor):
                out = out.cpu().numpy()
            v, a = float(out[0][0]), float(out[0][1])
            return v, a

        # try forward returning tensor-like [batch,2]
        try:
            result = regressor(input_ids=input_ids, attention_mask=att)
            # result might be tensor or tuple
            if isinstance(result, tuple) or isinstance(result, list):
                # prefer first element
                out = result[0]
            else:
                out = result
            if isinstance(out, torch.Tensor):
                out = out.cpu().numpy()
            v, a = float(out[0][0]), float(out[0][1])
            return v, a
        except Exception:
            pass

    # fallback default neutral
    return 5.00, 5.00

def run_pipeline(args):

    # --- load input
    data = load_jsonl(args.input_file)
    n = len(data)
    print(f"Loaded {n} examples from {args.input_file}")

    # --- prepare ICL generator
    icl_gen = ICLGenerator(model_name=args.gen_model, device=args.device)
    print(f"Loaded ICL generator {args.gen_model} on {args.device}")

    # --- load regressor from checkpoint (robust)
    # Instantiate model class with default args, then load weights
    reg_kwargs = {"model_name": args.reg_model_name} if args.reg_model_name else {}
    print(f"Loading regressor checkpoint from {args.reg_checkpoint} ...")
    regressor = build_regressor_from_checkpoint(args.reg_checkpoint, TransformerVARegressor, reg_kwargs, device=args.device)
    reg_tokenizer = AutoTokenizer.from_pretrained(args.reg_model_name, use_fast=True)

    # --- choose few-shot examples (prefer those in dataset with Quadruplet/non-null)
    examples_pool = [d for d in data if any(q.get("Aspect","").upper() != "NULL" and q.get("Opinion","").upper() != "NULL" for q in d.get("Quadruplet", []))]
    if not examples_pool:
        examples_pool = data
    random.shuffle(examples_pool)

    outputs = []
    for i, item in enumerate(data):
        text = item.get("Text", "").strip()
        id_ = item.get("ID", str(i))

        # sample examples for prompt (exclude the item itself if it has labels)
        examples = []
        for ex in examples_pool:
            if ex.get("ID") == id_:
                continue
            examples.append({
                "Text": ex.get("Text"),
                "Triplet": ex.get("Quadruplet") or ex.get("Triplet") or []
            })
            if len(examples) >= args.icl_examples:
                break

        # generate pairs
        pairs = icl_gen.generate_pairs(
            examples=examples,
            query_text=text,
            max_length=args.max_gen_len,
            temperature=args.temperature,
            top_k=args.top_k
        )

        # If generator returned nothing, fall back to empty list
        if not pairs:
            pairs = []

        triplets_out = []
        for asp, opn in pairs:
            # prefer opinion phrase for VA scoring
            phrase_for_reg = opn if args.score_opinion else f"{asp} {opn}"
            try:
                v, a = predict_va_with_regressor(regressor, reg_tokenizer, args.device, phrase_for_reg, max_len=args.reg_max_len)
            except Exception as e:
                print(f"Warning: regressor failed on phrase '{phrase_for_reg}': {e}")
                v, a = 5.0, 5.0
            triplets_out.append({
                "Aspect": asp,
                "Opinion": opn,
                "VA": f"{v:.2f}#{a:.2f}"
            })

        outputs.append({"ID": id_, "Triplet": triplets_out})
        if (i+1) % 50 == 0 or i == n-1:
            print(f"[{i+1}/{n}] processed. ID={id_}, triplets={len(triplets_out)}")

    save_json = args.output_file
    save_dir = os.path.dirname(save_json) or "."
    os.makedirs(save_dir, exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved predictions to", save_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task2: ICL extraction + regressor scoring")
    parser.add_argument("--input_file", type=str, required=True, help="JSONL file with ID and Text (may include Quadruplet for few-shot)")
    parser.add_argument("--output_file", type=str, required=True, help="Where to write JSONL predictions")
    parser.add_argument("--gen_model", type=str, default="t5-base", help="Sequence-to-sequence generator for ICL")
    parser.add_argument("--reg_checkpoint", type=str, required=True, help="Path to Subtask1 checkpoint (best_model.pt)")
    parser.add_argument("--reg_model_name", type=str, default="bert-base-multilingual-cased", help="Backbone name used by the regressor")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--icl_examples", type=int, default=6)
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument("--reg_max_len", type=int, default=128, help="Max length for regressor tokenizer")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--score_opinion", action="store_true", help="If set, regression will score only the opinion phrase; otherwise score 'aspect opinion' concatenation")
    args = parser.parse_args()

    run_pipeline(args)
