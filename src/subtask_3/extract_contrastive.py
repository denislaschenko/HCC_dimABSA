# =========================
# COLAB / PATH SETUP
# =========================

import sys
import os

PROJECT_ROOT = "/content/HCC_dimABSA"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================
# STANDARD IMPORTS
# =========================

import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from src.subtask_1.train_subtask1 import main as train_reg
from src.shared import config


# =========================
# DOMAIN-SPECIFIC CATEGORY MAP
# =========================

ENTITY_ATTRIBUTE_MAP = {
    "restaurant": {
        "ENTITY": [
            "RESTAURANT", "FOOD", "DRINKS", "AMBIENCE", "SERVICE", "LOCATION"
        ],
        "ATTRIBUTE": [
            "GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", "MISCELLANEOUS"
        ]
    },
    "laptop": {
        "ENTITY": [
            "LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU",
            "FANS_COOLING", "PORTS", "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES",
            "BATTERY", "GRAPHICS", "HARD_DISK", "MULTIMEDIA_DEVICES",
            "HARDWARE", "SOFTWARE", "OS", "WARRANTY", "SHIPPING",
            "SUPPORT", "COMPANY"
        ],
        "ATTRIBUTE": [
            "GENERAL", "PRICE", "QUALITY", "DESIGN_FEATURES",
            "OPERATION_PERFORMANCE", "USABILITY",
            "PORTABILITY", "CONNECTIVITY", "MISCELLANEOUS"
        ]
    },
    "hotel": {
        "ENTITY": [
            "HOTEL", "ROOMS", "FACILITIES", "ROOM_AMENITIES",
            "SERVICE", "LOCATION", "FOOD_DRINKS"
        ],
        "ATTRIBUTE": [
            "GENERAL", "PRICE", "COMFORT", "CLEANLINESS",
            "QUALITY", "DESIGN_FEATURES",
            "STYLE_OPTIONS", "MISCELLANEOUS"
        ]
    },
    "finance": {
        "ENTITY": [
            "MARKET", "COMPANY", "BUSINESS", "PRODUCT"
        ],
        "ATTRIBUTE": [
            "GENERAL", "SALES", "PROFIT",
            "AMOUNT", "PRICE", "COST"
        ]
    }
}

DOMAIN = config.DOMAIN.lower()
ENTITY_SET = ENTITY_ATTRIBUTE_MAP[DOMAIN]["ENTITY"]
ATTRIBUTE_SET = ENTITY_ATTRIBUTE_MAP[DOMAIN]["ATTRIBUTE"]

VALID_CATEGORIES = sorted(
    f"{e}#{a}" for e in ENTITY_SET for a in ATTRIBUTE_SET
)

# =========================
# CONTRASTIVE CATEGORY MODEL
# =========================

CAT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
cat_model = SentenceTransformer(CAT_MODEL_NAME)

CATEGORY_EMBS = cat_model.encode(
    VALID_CATEGORIES,
    normalize_embeddings=True,
    convert_to_tensor=True
)

def contrastive_category(aspect: str, opinion: str, sentence: str) -> str:
    query = f"{aspect} {opinion} {sentence}"
    q_emb = cat_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_tensor=True
    )
    scores = util.cos_sim(q_emb, CATEGORY_EMBS)
    idx = int(torch.argmax(scores))
    return VALID_CATEGORIES[idx]


# =========================
# ICL PROMPT (Aspect + Opinion ONLY)
# =========================

ICL_PROMPT = """
Below is an instruction describing a task.

### Instruction:
Given a text, extract all (Aspect, Opinion) pairs.

- Aspect: exact span from the text
- Opinion: opinion expression describing the aspect

### Example:
Input:
{"ID": "lap1", "Text": "The battery life is excellent but the keyboard feels cheap"}

Output:
{"ID": "lap1", "Quadruplet": [
  {"Aspect": "battery life", "Opinion": "excellent", "VA": "0#0"},
  {"Aspect": "keyboard", "Opinion": "cheap", "VA": "0#0"}
]}

### Question:
Now complete the following example.
Never change the output layout.
Always predict VA as "0#0".
"""

def build_prompt(text: str, id_: str) -> str:
    return f"""<|user|>
{ICL_PROMPT}
Input:
{{"ID": "{id_}", "Text": "{text}"}}

Output:
<|assistant|>
"""


# =========================
# LLM OUTPUT PARSING
# =========================

def extract_quads_from_llm_output(output_text: str):
    try:
        data = json.loads(output_text)
        return [
            (q["Aspect"], q["Opinion"])
            for q in data.get("Quadruplet", [])
        ]
    except Exception:
        return []


# =========================
# MAIN PROCESSING
# =========================

def process_jsonl(model_name, input_path, output_path):
    print(f"Loading LLM: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total_lines = sum(1 for _ in open(input_path, encoding="utf-8"))
    print(f"Processing {total_lines} samples")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total_lines, desc="Extracting"):
            if not line.strip():
                continue

            item = json.loads(line)
            text = item["Text"]
            sample_id = item["ID"]

            prompt = build_prompt(text, sample_id)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id
                )

            gen_text = tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            pairs = extract_quads_from_llm_output(gen_text)

            formatted = []
            for aspect, opinion in pairs:
                category = contrastive_category(aspect, opinion, text)
                formatted.append({
                    "Aspect": aspect.strip(),
                    "Category": category,
                    "Opinion": opinion.strip(),
                    "VA": "0#0"
                })

            fout.write(json.dumps({
                "ID": sample_id,
                "Quadruplet": formatted
            }, ensure_ascii=False) + "\n")

    print("LLM extraction finished.")
    print("Running VA regressor...")
    train_reg(output_path)


# =========================
# ENTRY POINT
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
    )
    args = parser.parse_args()

    process_jsonl(
        args.model,
        config.PREDICT_FILE,
        config.PREDICTION_FILE
    )


if __name__ == "__main__":
    main()

