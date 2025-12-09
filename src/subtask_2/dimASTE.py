
"""
dimASTE.py

Single-file script to train, infer, and evaluate Dimensional Aspect Sentiment Triplet/Quadruplet extraction

NOTE: This script follows the Colab notebook but organizes it in a single runnable file.
"""



import os
import re
import json
import argparse
import zipfile
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import math
from datetime import datetime
import torch
pip install unsloth
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer



try:
    from unsloth import FastLanguageModel
except Exception as e:
    raise RuntimeError("Missing `unsloth`. Install it or run in an environment similar to the Colab. Error: " + str(e))

from transformers import TrainingArguments
from trl import SFTTrainer

# -----------------------
# Default configuration
# -----------------------
DEFAULT_MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
MAX_SEQ_LEN = 512

# -----------------------
# Helpers: extraction & metrics
# -----------------------

def extract_triplets_from_text(text: str, task: str = "task2") -> List[Dict]:
    """
    Extracts triplets or quadruplets from a model-generated string.
    Accepts common variants of the model output and returns cleaned list of dicts.

    For task2: returns [{"Aspect":..., "Opinion":..., "VA": "V#A"}, ...]
    For task3: returns [{"Aspect":..., "Category":..., "Opinion":..., "VA":...}, ...]
    """
    results = []
    if task == "task2":
        # allow patterns like (aspect, opinion, 6.75#6.38)
        pattern = r'\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([0-9]+\.[0-9]{1,}#[0-9]+\.[0-9]{1,})\s*\)'
        matches = re.findall(pattern, text)
        for aspect, opinion, va in matches:
            results.append({
                "Aspect": aspect.strip(),
                "Opinion": opinion.strip(),
                "VA": normalize_va_string(va.strip())
            })
    elif task == "task3":
        # pattern: (aspect, CATEGORY#ATTRIBUTE, opinion, V#A)
        pattern = r'\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([0-9]+\.[0-9]{1,}#[0-9]+\.[0-9]{1,})\s*\)'
        matches = re.findall(pattern, text)
        for aspect, category, opinion, va in matches:
            results.append({
                "Aspect": aspect.strip(),
                "Category": category.strip(),
                "Opinion": opinion.strip(),
                "VA": normalize_va_string(va.strip())
            })
    else:
        raise ValueError("task must be 'task2' or 'task3'")
    return results

def normalize_va_string(va: str) -> str:
    """
    Ensure VA string is formatted V.VV#A.AA and clipped to [1.00, 9.00].
    Accepts inputs like 6.75#6.38 or 6.7#6.4 and returns 6.70#6.40.
    """
    try:
        v_str, a_str = va.split("#")
        v = float(v_str)
        a = float(a_str)
    except Exception:
        # fallback: return as-is (we'll try to be robust)
        return va
    v = max(1.0, min(9.0, v))
    a = max(1.0, min(9.0, a))
    return f"{v:.2f}#{a:.2f}"

def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(items: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def zip_folder(folder_path: str, output_zip: str):
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full = os.path.join(root, file)
                arcname = os.path.relpath(full, start=os.path.dirname(folder_path))
                zf.write(full, arcname)

# -----------------------
# Evaluation metrics
# -----------------------

def triplet_key(d: Dict, task: str = "task2") -> str:
    """Key used for exact matching of triplets/quadruplets.
    For task2: Aspect + '||' + Opinion
    For task3: Aspect + '||' + Category + '||' + Opinion
    """
    if task == "task2":
        return f"{d['Aspect']}||{d['Opinion']}"
    else:
        return f"{d['Aspect']}||{d['Category']}||{d['Opinion']}"

def compute_extraction_f1(gold_items: List[Dict], pred_items: List[Dict], task: str="task2") -> Tuple[float, float, float, int, int, int]:
    """
    Exact-match F1 on Aspect+Opinion (and Category if task3).
    Returns: precision, recall, f1, tp, fp, fn
    """
    gold_keys = set([triplet_key(g, task) for g in gold_items])
    pred_keys = set([triplet_key(p, task) for p in pred_items])

    tp_keys = gold_keys & pred_keys
    fp_keys = pred_keys - gold_keys
    fn_keys = gold_keys - pred_keys

    tp = len(tp_keys)
    fp = len(fp_keys)
    fn = len(fn_keys)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp, fp, fn

def parse_va(va_str: str) -> Tuple[float,float]:
    """Return (valence, arousal) floats from 'V.VV#A.AA' or best-effort parsing."""
    try:
        v, a = va_str.split("#")
        return float(v), float(a)
    except Exception:
        # fallback zero
        return 0.0, 0.0

def compute_va_errors(gold_items: List[Dict], pred_items: List[Dict], task: str="task2") -> Tuple[float,float,int]:
    """
    For matched triplets (exact match on A+O (+Category)), compute MAE and RMSE for valence only (and optionally arousal).
    Returns (MAE_valence, RMSE_valence, n_matched)
    """
    gold_map = {triplet_key(g, task): g for g in gold_items}
    pred_map = {triplet_key(p, task): p for p in pred_items}
    matched_keys = set(gold_map.keys()) & set(pred_map.keys())

    if not matched_keys:
        return 0.0, 0.0, 0
    errs = []
    for k in matched_keys:
        g_va = gold_map[k].get("VA", "")
        p_va = pred_map[k].get("VA", "")
        g_v, _ = parse_va(g_va)
        p_v, _ = parse_va(p_va)
        errs.append(abs(g_v - p_v))
    n = len(errs)
    mae = sum(errs) / n
    rmse = math.sqrt(sum(e*e for e in errs) / n)
    return mae, rmse, n

# -----------------------
# Prompt templates
# -----------------------

def build_instruction(task: str, domain: str, lang: str) -> str:
    """Return instruction string for task2 or task3 (mirrors the notebook)."""
    if task == "task2":
        return '''Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, O, VA) triplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

Valence ranges from 1 (negative) to 9 (positive),
Arousal ranges from 1 (calm) to 9 (excited).

### Example:
Input:
[Text] average to good thai food, but terrible delivery.

Output:
[Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)

### Question:
Now complete the following example:
Input:
'''
    else:
        # Minimal category hints for common domains
        map_entity = {
            'restaurant': ('RESTAURANT, FOOD, DRINKS, AMBIENCE, SERVICE, LOCATION',
                           'GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS'),
            'laptop': ('LAPTOP, DISPLAY, KEYBOARD, MOUSE, MOTHERBOARD, CPU, FANS_COOLING, PORTS, MEMORY, POWER_SUPPLY, OPTICAL_DRIVES, BATTERY, GRAPHICS, HARD_DISK, MULTIMEDIA_DEVICES, HARDWARE, SOFTWARE, OS, WARRANTY, SHIPPING, SUPPORT, COMPANY',
                       'GENERAL, PRICE, QUALITY, DESIGN_FEATURES, OPERATION_PERFORMANCE, USABILITY, PORTABILITY, CONNECTIVITY, MISCELLANEOUS')
        }
        ent, att = map_entity.get(domain, ("", ""))
        return f'''Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, C, O, VA) quadruplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- C is a Category label (e.g. FOOD#QUALITY)
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

Valence ranges from 1 (negative) to 9 (positive),
Arousal ranges from 1 (calm) to 9 (excited).

### Label constraints:
[Entity Labels] ({ent})
[Attribute Labels] ({att})

### Example:
Input:
[Text] average to good thai food, but terrible delivery.

Output:
[Quadruplet] (thai food, FOOD#QUALITY, average to good, 6.75#6.38),
             (delivery, SERVICE#GENERAL, terrible, 2.88#6.62)

### Question:
Now complete the following example:
Input:
'''

# -----------------------
# Dataset conversion for training
# -----------------------
def convert_example_for_training(ex: Dict, instruction: str, task: str) -> Dict:
    """
    Format each example into the chat-style prompt as used in the notebook:
    <|user|>\n[instruction + Text]\n<|assistant|>\n[gold answer]
    The trainer expects a "text" field containing the whole pair.
    """
    text = ex["Text"]
    quads = ex.get("Quadruplet", [])
    if task == "task2":
        answer = ", ".join([f"({q['Aspect']}, {q['Opinion']}, {q['VA']})" for q in quads])
    else:
        answer = ", ".join([f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})" for q in quads])
    prompt = instruction + "[Text] " + text + "\n\nOutput:"
    return {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}"}

# -----------------------
# Model creation (unsloth + LoRA)
# -----------------------
def build_model_and_tokenizer(model_id: str = DEFAULT_MODEL_ID, max_seq_length: int = MAX_SEQ_LEN):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return model, tokenizer

# -----------------------
# Training wrapper
# -----------------------
def train(args):
    subtask = args.subtask
    task = args.task
    lang = args.lang
    domain = args.domain

    train_url = f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/{subtask}/{lang}/{lang}_{domain}_train_alltasks.jsonl"
    print("Loading training data from:", train_url)
    dataset = load_dataset("json", data_files=train_url)["train"]
    print("Total examples loaded:", len(dataset))

    instruction = build_instruction(task=task, domain=domain, lang=lang)
    dataset = dataset.map(lambda ex: convert_example_for_training(ex, instruction, task))
    # split into train/val
    split = dataset.train_test_split(test_size=args.val_split, shuffle=True)
    train_set = split["train"]
    val_set = split["test"]
    print(f"Train size: {len(train_set)}  |  Val size: {len(val_set)}")

    # build model
    model, tokenizer = build_model_and_tokenizer(model_id=args.model_id, max_seq_length=args.max_seq_len)

    # Trainer / TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=20,
        logging_steps=50,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True if args.fp16 else False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_set,
        eval_dataset=val_set,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
        packing=False,
    )

    # train
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training completed. Model and tokenizer saved to:", args.output_dir)

# -----------------------
# Inference wrapper
# -----------------------
def infer(args):
    subtask = args.subtask
    task = args.task
    lang = args.lang
    domain = args.domain

    instruction = build_instruction(task=task, domain=domain, lang=lang)

    # load model
    model, tokenizer = build_model_and_tokenizer(model_id=args.model_id, max_seq_length=args.max_seq_len)
    # load LoRA weights if they exist in output_dir
    if os.path.isdir(args.output_dir):
        try:
            print("Loading LoRA / adapter from", args.output_dir)
            model.load_adapter(args.output_dir)
        except Exception:
            # unsloth may provide different method; ignore if not needed
            pass

    # load predict dataset (either URL or local file)
    predict_data = None
    if args.predict_url:
        print("Loading predict dataset:", args.predict_url)
        try:
            predict_data = load_dataset("json", data_files=args.predict_url)["train"]
        except Exception:
            # try local jsonl
            predict_data = load_dataset("json", data_files={"train": args.predict_url})["train"]
    else:
        raise ValueError("predict_url must be specified for inference mode")

    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    for i, sample in enumerate(predict_data):
        text = sample["Text"]
        final_prompt = instruction + "[Text] " + text + "\n\nOutput:"
        # wrap in chat tokens as in training
        messages = f"<|user|>\n{final_prompt}\n<|assistant|>\n"
        # apply template if tokenizer supports it (not all do)
        try:
            tokenized = tokenizer.apply_chat_template([{"role":"user","content": final_prompt}], tokenize=False, add_generation_prompt=True)
            # tokenized is text (string)
            input_text = tokenized
        except Exception:
            input_text = messages

        # generate
        gen = model.generate(
            **tokenizer(input_text, return_tensors="pt").to("cuda"),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        decoded = tokenizer.decode(gen[0])
        # The generated decoded text often contains prior messages; try to extract last assistant reply
        # We attempt to take everything after the last newline for simplicity then fall back.
        # Better heuristics could be added.
        extracted_text = decoded.split("\n")[-1].strip()
        # parse triplets
        parsed = extract_triplets_from_text(decoded, task="task2" if task=="task2" else "task3")
        key_name = "Triplet" if task == "task2" else "Quadruplet"
        dump = {
            "ID": sample.get("ID", f"sample_{i}"),
            "Text": text,
            key_name: parsed
        }
        results.append(dump)
        if (i+1) % 50 == 0:
            print(f"Inferred {i+1}/{len(predict_data)}")

    # write jsonl to output_dir
    out_name = f"pred_{lang}_{domain}_{task}.jsonl"
    out_path = os.path.join(args.output_dir, out_name)
    write_jsonl(results, out_path)
    print("Saved predictions to:", out_path)

    # zip output folder for submission convenience
    zip_name = os.path.join(args.output_dir, f"submission_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.zip")
    zip_folder(args.output_dir, zip_name)
    print("Zipped output folder to:", zip_name)

# -----------------------
# Evaluation wrapper
# -----------------------
def evaluate(args):
    gold = read_jsonl(args.gold)
    pred = read_jsonl(args.pred)

    # Map by ID for per-instance comparison
    pred_map = {item["ID"]: item for item in pred}
    gold_map = {item["ID"]: item for item in gold}

    # Support both Triplet and Quadruplet key naming
    task = args.task
    key_name = "Triplet" if task == "task2" else "Quadruplet"

    total_tp = total_fp = total_fn = 0
    sum_prec = sum_rec = sum_f1 = 0.0
    total_mae = 0.0
    total_rmse_numer = 0.0
    total_matched_va = 0

    per_instance_results = []

    for gid, gold_item in gold_map.items():
        gold_trips = gold_item.get(key_name, [])
        pred_item = pred_map.get(gid, {"ID": gid, key_name: []})
        pred_trips = pred_item.get(key_name, [])

        prec, rec, f1, tp, fp, fn = compute_extraction_f1(gold_trips, pred_trips, task="task2" if task == "task2" else "task3")
        mae, rmse, n_matched = compute_va_errors(gold_trips, pred_trips, task="task2" if task == "task2" else "task3")

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_mae += mae * n_matched
        total_rmse_numer += (rmse**2) * n_matched
        total_matched_va += n_matched

        per_instance_results.append({
            "ID": gid,
            "prec": prec, "rec": rec, "f1": f1, "va_mae": mae, "va_rmse": rmse, "n_matched": n_matched
        })

    # Corpus-level metrics
    corpus_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    corpus_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    corpus_f1 = (2*corpus_prec*corpus_rec/(corpus_prec+corpus_rec)) if (corpus_prec+corpus_rec)>0 else 0.0

    corpus_mae = total_mae / total_matched_va if total_matched_va > 0 else 0.0
    corpus_rmse = math.sqrt(total_rmse_numer / total_matched_va) if total_matched_va > 0 else 0.0

    results = {
        "corpus_precision": corpus_prec,
        "corpus_recall": corpus_rec,
        "corpus_f1": corpus_f1,
        "va_mae_on_matched": corpus_mae,
        "va_rmse_on_matched": corpus_rmse,
        "total_matched_va_pairs": total_matched_va,
    }

    print("Evaluation results:")
    print(json.dumps(results, indent=2))
    # Optionally save detailed per-instance results
    if args.output:
        write_jsonl(per_instance_results, args.output)
        print("Saved per-instance metrics to:", args.output)

# -----------------------
# CLI: parse args & dispatch
# -----------------------
def get_common_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--subtask", type=str, default="subtask_2", help="subtask_2 or subtask_3")
    p.add_argument("--task", type=str, default="task2", help="task2 or task3")
    p.add_argument("--lang", type=str, default="eng")
    p.add_argument("--domain", type=str, default="restaurant")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--output_dir", type=str, default="./output_lora")
    p.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    return p

def main():
    parser = argparse.ArgumentParser(description="dimASTE train / infer / eval")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # train
    p_train = subparsers.add_parser("train", parents=[get_common_parser()], help="Train model")
    p_train.add_argument("--batch_size", type=int, default=1)
    p_train.add_argument("--grad_accum", type=int, default=4)
    p_train.add_argument("--epochs", type=int, default=2)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--val_split", type=float, default=0.1)
    p_train.add_argument("--save_steps", type=int, default=200)
    p_train.add_argument("--eval_steps", type=int, default=200)
    p_train.add_argument("--fp16", action="store_true")
    p_train.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p_train.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)

    # infer
    p_infer = subparsers.add_parser("infer", parents=[get_common_parser()], help="Run inference")
    p_infer.add_argument("--predict_url", type=str, required=True, help="URL or local path to predict jsonl")
    p_infer.add_argument("--temperature", type=float, default=0.0)
    p_infer.add_argument("--top_p", type=float, default=1.0)
    p_infer.add_argument("--top_k", type=int, default=50)
    p_infer.add_argument("--max_new_tokens", type=int, default=512)
    p_infer.add_argument("--batch_size", type=int, default=1)
    p_infer.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    p_infer.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate predictions vs gold")
    p_eval.add_argument("--gold", type=str, required=True, help="Gold JSONL path")
    p_eval.add_argument("--pred", type=str, required=True, help="Predictions JSONL path")
    p_eval.add_argument("--task", type=str, default="task2", choices=["task2","task3"])
    p_eval.add_argument("--output", type=str, default=None, help="Optional per-instance metrics JSONL output")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        raise ValueError("Unknown mode: " + str(args.mode))

if __name__ == "__main__":
    main()
