import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import re
import argparse
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
import difflib

from src.shared import config

try:
    from src.subtask_1.train_subtask1_clean import predict_va_for_subtask2
except ImportError:
    print("Warning: Could not import predict_va_for_subtask2. Regression step will fail.")


INSTRUCTION = """Given a textual instance [Text], extract all (A, O, VA) triplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

RULES:
1. EXTRACT EXACTLY: Copy the substring exactly as it appears in the text, including typos and capitalization.
2. NO NULLS: Every triplet must have explicit Aspects and Opinions from the text. Do not generate 'NULL'.
3. NO HALLUCINATIONS: If a word is not in the text, do not extract it.
4. VA FORMAT: Predict the specific Valence#Arousal score (e.g., 7.5#4.2).

Be exhaustive: extract every aspect and opinion mentioned. You must preserve the EXACT capitalization and spelling as it appears in the [Text]."""

ROBUST_PATTERN = r'\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)'

class ExampleRetriever:
    """RAG Retriever for finding similar examples in the training set."""

    def __init__(self, train_path, model_name='BAAI/bge-base-en-v1.5'):
        print(f"Loading retriever model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.examples = []

        if not os.path.exists(train_path):
            print(f"Warning: Training file for RAG not found at {train_path}")
            return

        with open(train_path, 'r', encoding='UTF-8') as f:
            raw_data = [json.loads(line) for line in f if line.strip()]

        for item in raw_data:
            quads = item.get("Quadruplet", [])
            valid_triplets = [f"({q['Aspect']}, {q['Opinion']}, {q['VA']})"
                              for q in quads if q.get("Aspect") != "NULL"]

            if valid_triplets:
                self.examples.append({
                    "Text": item["Text"],
                    "Answer": ", ".join(valid_triplets)
                })

        if self.examples:
            texts = [ex["Text"] for ex in self.examples]
            self.embeddings = self.encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        else:
            self.embeddings = None

    def retrieve(self, query_text, k=3):
        if self.embeddings is None: return []
        query_emb = self.encoder.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_results = torch.topk(scores, k=min(k, len(self.examples)))
        return [self.examples[idx] for idx in top_results.indices]


def build_infer_prompt(tokenizer, text, retrieved_examples):
    messages = [{"role": "system", "content": INSTRUCTION}]
    for ex in retrieved_examples:
        messages.append({"role": "user", "content": f"Extract triplets: {ex['Text']}"})
        messages.append({"role": "assistant", "content": ex["Answer"]})
    messages.append({"role": "user", "content": f"[Text] {text}\n\nOutput:\n"})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_triplets_regex(decoded_text):
    pattern = r'\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)'
    matches = re.findall(pattern, decoded_text)
    return [{"Aspect": a.strip(), "Opinion": o.strip(), "VA": v.replace(" ", "")} for a, o, v in matches]


def parse_and_fix_triplet(match_tuple):
    """
    Takes raw strings like (' Screen ', ' good', '8.5# 4.2 ')
    Returns a clean dict or None if totally broken.
    """
    aspect, opinion, va_raw = match_tuple

    aspect = aspect.strip()
    opinion = opinion.strip()

    final_va = "0#0"

    try:
        clean_nums = "".join([c for c in va_raw if c.isdigit() or c in ".#"])

        if "#" in clean_nums:
            parts = clean_nums.split("#")
            if len(parts) == 2 and parts[0] and parts[1]:
                v = float(parts[0])
                a = float(parts[1])
                # Clip to legal range [1, 9] to be safe
                v = max(1.0, min(9.0, v))
                a = max(1.0, min(9.0, a))
                final_va = f"{v:.2f}#{a:.2f}"
    except:
        pass

    return {"Aspect": aspect, "Opinion": opinion, "VA": final_va}

def ground_to_text_fuzzy(text, phrase, threshold=0.85):
    if not phrase or phrase == "NULL": return None
    lower_text = text.lower()
    lower_phrase = phrase.lower()

    if lower_phrase in lower_text:
        start = lower_text.find(lower_phrase)
        return text[start:start + len(phrase)]

    best_ratio = 0
    best_match = None
    n = len(phrase)
    for length in range(max(1, n - 2), min(len(text), n + 3)):
        for i in range(len(text) - length + 1):
            window = text[i: i + length]
            ratio = difflib.SequenceMatcher(None, lower_phrase, window.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = window

    return best_match if best_ratio > threshold else None


def ground_triplet(text, triplet):
    ra = ground_to_text_fuzzy(text, triplet['Aspect'])
    ro = ground_to_text_fuzzy(text, triplet['Opinion'])

    if ra and ro and ra.lower() != ro.lower():
        triplet['Aspect'] = ra
        triplet['Opinion'] = ro
        return triplet
    return None


def sanitize_va(va_str):
    try:
        v_str, a_str = va_str.split('#')
        v, a = float(v_str), float(a_str)
        v = max(1.0, min(9.0, v))
        a = max(1.0, min(9.0, a))
        return f"{v:.2f}#{a:.2f}"
    except ValueError:
        return "5.00#5.00"

def run_inference(checkpoint_path, input_file, output_file, base_model_id="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"):
    print(f"\n{'=' * 20}\nStarting Inference\nCheckpoint: {checkpoint_path}\nInput: {input_file}\n{'=' * 20}")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    train_file = os.path.join(os.path.dirname(input_file), f"{config.LANG}_{config.DOMAIN}_production_train.jsonl")

    if not os.path.exists(train_file):
        train_file = config.TRAIN_FILE

    retriever = ExampleRetriever(train_file)
    RAG_K = config.current_config.get("RAG_K")

    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        for line in tqdm(lines, desc="Generating"):
            item = json.loads(line)

            dynamic_examples = retriever.retrieve(item["Text"], k=RAG_K)
            prompt = build_infer_prompt(tokenizer, item["Text"], dynamic_examples)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    # top_p=0.9,
                    # num_return_sequences=N_VOTES,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            matches = re.findall(ROBUST_PATTERN, decoded)
            grounded_triplets = []

            for m in matches:
                clean_triplet = parse_and_fix_triplet(m)
                gt = ground_triplet(item["Text"], clean_triplet)
                if gt:
                    grounded_triplets.append(gt)

            record = {"ID": item["ID"], "Text": item["Text"], "Triplet": grounded_triplets}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Inference complete. Results saved to {output_file}")

    print(f"\nExample generation complete. Handing over to Regression Model...")

    extracted_data = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                extracted_data.append(json.loads(line))

    subtask1_model_dir = "/workspace/HCC_dimABSA_remote/outputs/subtask_1/models"

    if not os.path.exists(subtask1_model_dir):
        print(f"CRITICAL WARNING: Model dir {subtask1_model_dir} not found! Keeping LLM scores.")
    else:
        final_results = predict_va_for_subtask2(extracted_data, model_dir=subtask1_model_dir)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Regression complete. {len(final_results)} predictions updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, default=config.PREDICT_FILE)
    parser.add_argument("--output", type=str,
                        default=os.path.join(config.PROJECT_ROOT, "outputs", "subtask_2", "predictions",
                                             f"pred_{config.LANG}_{config.DOMAIN}.jsonl"))
    args = parser.parse_args()

    run_inference(args.checkpoint, args.input, args.output)