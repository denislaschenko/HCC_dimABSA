import difflib
import sys
import os
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
from src.subtask_1.train_subtask1_ens import predict_va_for_subtask2 as regression

# Path setup
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared import config
from src.subtask_1.train_subtask1_ens import predict_va_for_subtask2

# --- CONFIGURATION ---
BASE_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
SUBTASK_NAME = "subtask_3"

INSTRUCTION = """Given a textual instance [Text], extract all (A, O, VA) triplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

RULES:
1. EXTRACT EXACTLY: Copy the substring exactly as it appears in the text, including typos and capitalization.
2. NO NULLS: Every triplet must have explicit Aspects and Opinions from the text. Do not generate 'NULL'.
3. NO HALLUCINATIONS: If a word is not in the text, do not extract it.
4. VA FORMAT: Use '0#0' as a placeholder for the VA score; it will be calculated by a secondary model.

Be exhaustive: extract every aspect and opinion mentioned. You must preserve the EXACT capitalization and spelling as it appears in the [Text]."""

def extract_triplets_regex(decoded_text):
    # Pattern captures (Aspect, Opinion, and optionally VA)
    pattern = r'\((.*?), (.*?)(?:, (\d+(?:\.\d+)?\s*#\s*\d+(?:\.\d+)?))?\)'
    matches = re.findall(pattern, decoded_text)
    # Default to 0#0 if the LLM omits the VA field
    return [{"Aspect": a.strip(), "Opinion": o.strip(), "VA": v.replace(" ", "") if v else "0#0"} for a, o, v in matches]

class ExampleRetriever:
    def __init__(self, train_path, model_name='BAAI/bge-large-en-v1.5'):
        print(f"Loading retriever model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.examples = []

        print("Reading training data...")
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

        # --- FIX: Encode ONLY ONCE after the loop finishes ---
        print(f"Encoding {len(self.examples)} examples...")
        texts = [ex["Text"] for ex in self.examples]
        self.embeddings = self.encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def retrieve(self, query_text, k=8):
        query_emb = self.encoder.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_results = torch.topk(scores, k=min(k, len(self.examples)))
        indices = top_results.indices.flip(dims=[0])
        return [self.examples[idx] for idx in indices]


def build_infer_prompt(tokenizer, text, retrieved_examples):
    messages = [{"role": "system", "content": INSTRUCTION}]
    for ex in retrieved_examples:
        messages.append({"role": "user", "content": f"Extract triplets: {ex['Text']}"})
        messages.append({"role": "assistant", "content": ex["Answer"]})
    messages.append({"role": "user", "content": f"[Text] {text}\n\nOutput:\n"})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def get_all_indices(text, phrase):
    """Returns list of all indices where phrase appears."""
    if not phrase: return []
    indices = []
    lower_text = text.lower()
    lower_phrase = phrase.lower()
    start = 0
    while True:
        idx = lower_text.find(lower_phrase, start)
        if idx == -1: break
        indices.append((idx, text[idx : idx + len(phrase)]))
        start = idx + 1
    return indices


def ground_to_text_fuzzy(text, phrase, threshold=0.85):
    """Restored Fuzzy Logic using difflib"""
    if not phrase or phrase == "NULL": return None
    lower_text = text.lower()
    lower_phrase = phrase.lower()

    # Fuzzy Match Search
    best_ratio = 0
    best_match = None
    n = len(phrase)

    for length in range(max(1, n - 3), min(len(text), n + 4)):
        for i in range(len(text) - length + 1):
            window = text[i: i + length]
            if window and lower_phrase and (
                    window[0].lower() != lower_phrase[0] and window[-1].lower() != lower_phrase[-1]):
                continue

            ratio = difflib.SequenceMatcher(None, lower_phrase, window.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = window

    if best_ratio > threshold:
        return best_match
    return None


def ground_with_proximity(text, raw_aspect, raw_opinion):
    # 1. Try Exact Matches
    aspect_matches = get_all_indices(text, raw_aspect)
    opinion_matches = get_all_indices(text, raw_opinion)

    # 2. Fallback: If exact match fails, use FUZZY match
    if not aspect_matches or not opinion_matches:
        ra = ground_to_text_fuzzy(text, raw_aspect)
        ro = ground_to_text_fuzzy(text, raw_opinion)
        return ra, ro

    # 3. Proximity Logic (if multiple exact matches found)
    best_a = None
    best_o = None
    min_dist = float('inf')

    for a_idx, a_val in aspect_matches:
        for o_idx, o_val in opinion_matches:
            dist = abs(a_idx - o_idx)
            if dist < min_dist:
                min_dist = dist
                best_a = a_val
                best_o = o_val

    return best_a, best_o


def run_majority_vote(candidates_list, threshold):
    vote_counter = defaultdict(int)
    va_accumulator = defaultdict(list)
    raw_map = {}

    for run in candidates_list:
        seen_in_run = set()
        for t in run:
            key = (t['Aspect'].lower(), t['Opinion'].lower())
            if key in seen_in_run: continue
            seen_in_run.add(key)
            vote_counter[key] += 1
            try:
                v, a = map(float, t['VA'].split('#'))
                va_accumulator[key].append((v, a))
            except:
                continue

            if key not in raw_map:
                raw_map[key] = {'Aspect': t['Aspect'], 'Opinion': t['Opinion']}

    final_triplets = []
    for key, count in vote_counter.items():
        if count >= threshold:
            scores = np.mean(va_accumulator[key], axis=0)
            winner = raw_map[key]
            winner['VA'] = f"{scores[0]:.2f}#{scores[1]:.2f}"
            final_triplets.append(winner)
    return final_triplets


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # Paths
    DATA_DIR = os.path.join(config.PROJECT_ROOT, "task-dataset", "track_a", SUBTASK_NAME, config.LANG)
    PREDICTION_DIR = os.path.join(config.PROJECT_ROOT, "outputs", SUBTASK_NAME, "predictions")
    TRAIN_FILE = os.path.join(DATA_DIR, f"{config.LANG}_{config.DOMAIN}_production_train.jsonl")
    PREDICT_FILE = os.path.join(DATA_DIR, f"{config.LANG}_{config.DOMAIN}_test_task3.jsonl")
    output_file = os.path.join(PREDICTION_DIR, f"pred_{config.LANG}_{config.DOMAIN}.jsonl")

    # Load Model
    retriever = ExampleRetriever(TRAIN_FILE)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto",
                                                 trust_remote_code=True)

    # Fix Tokenizer Initialization
    print(f"Loading Tokenizer: {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print(f"Loading Adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()


    N_VOTES = 1
    TEMP = 0.1
    THRESHOLD = 1
    RAG_K = 8

    print(f"Running HIGH RECALL Ensemble: {N_VOTES} Votes, Threshold={THRESHOLD}, Fuzzy Grounding...")

    with open(PREDICT_FILE, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Voting"):
            item = json.loads(line)

            dynamic_examples = retriever.retrieve(item["Text"], k=RAG_K)
            prompt = build_infer_prompt(tokenizer, item["Text"], dynamic_examples)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=TEMP,
                    top_p=0.9,
                    num_return_sequences=N_VOTES,
                    pad_token_id=tokenizer.eos_token_id
                )

            candidates = []
            for i in range(N_VOTES):
                decoded = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                raw = extract_triplets_regex(decoded)
                grounded = []
                for t in raw:
                    ra, ro = ground_with_proximity(item["Text"], t['Aspect'], t['Opinion'])

                    if ra and ro and ra.lower() != ro.lower():
                        t['Aspect'] = ra
                        t['Opinion'] = ro
                        grounded.append(t)

                candidates.append(grounded)

            final_triplets = run_majority_vote(candidates, threshold=THRESHOLD)

            fout.write(json.dumps({"ID": item["ID"], "Text": item["Text"], "Triplet": final_triplets},
                                  ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nFinished. Passing to regression model...")

    # extracted_data = []
    # with open(output_file, "r", encoding="utf-8") as f:
    #     for line in f:
    #         if line.strip():
    #             extracted_data.append(json.loads(line))
    #
    # print(f"\nFinished. Passing to regression model...")
    #
    # # --- FIX: Point to Subtask 1 models explicitly ---
    # # The regression models are in outputs/subtask_1/models, not subtask_2
    # subtask1_model_dir = os.path.join(config.PROJECT_ROOT, "outputs", "subtask_1", "models")
    #
    # # Check if the directory exists to avoid confusion
    # if not os.path.exists(subtask1_model_dir):
    #     print(f"Warning: Subtask 1 model directory not found at {subtask1_model_dir}")
    #
    # # Pass the model_dir argument to the function
    # # Note: Ensure predict_va_for_subtask2 in src/subtask_1/train_subtask1_ens.py accepts 'model_dir'
    # final_results = regression(extracted_data, model_dir=subtask1_model_dir)
    #
    # # with open(output_file, "w", encoding="utf-8") as f:

    # print(f"Done! Saved {len(final_results)} predictions to {output_file}")

if __name__ == "__main__":
    main()