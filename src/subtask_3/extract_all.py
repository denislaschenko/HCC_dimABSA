import json
import argparse
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.subtask_1.train_subtask1 import main as train_reg
from src.shared import config
import sys
import os

# --- COLAB-SAFE PATH FIX ---
PROJECT_ROOT = "/content/HCC_dimABSA"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.subtask_1.train_subtask1 import main as train_reg

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

VALID_CATEGORIES = {
    f"{e}#{a}" for e in ENTITY_SET for a in ATTRIBUTE_SET
}


# =========================
# ICL PROMPT
# =========================

ICL_PROMPT = f"""
Below is an instruction describing a task.

### Instruction:
Given a text, extract all (A, C, O) quadruplets:
- A: Aspect term (exact span from text)
- C: Aspect Category in ENTITY#ATTRIBUTE format (UPPERCASE)
- O: Opinion term

Only use categories from the following list:
{sorted(list(VALID_CATEGORIES))}

### Example:
Input:
{{"ID": "lap1", "Text": "The battery life is excellent but the keyboard feels cheap"}}

Output:
{{"ID": "lap1", "Quadruplet": [
  {{"Aspect": "battery life", "Category": "BATTERY#QUALITY", "Opinion": "excellent", "VA": "0#0"}},
  {{"Aspect": "keyboard", "Category": "KEYBOARD#QUALITY", "Opinion": "cheap", "VA": "0#0"}}
]}}

### Question:
Now complete the following example.
Never change the output layout.
Always predict VA as "0#0".
"""


def build_prompt(text: str, id_: str):
    return f"""<|user|>
{ICL_PROMPT}
Input:
{{"ID": "{id_}", "Text": "{text}"}}

Output:
<|assistant|>
"""


# =========================
# OUTPUT PARSING (JSON SAFE)
# =========================

def extract_quads_from_llm_output(output_text: str):
    try:
        data = json.loads(output_text)
        return [
            (q["Aspect"], q["Category"], q["Opinion"])
            for q in data.get("Quadruplet", [])
        ]
    except Exception:
        return []


def normalize_category(category: str):
    category = category.upper().replace(" ", "")
    if category in VALID_CATEGORIES:
        return category

    if "#" in category:
        ent, attr = category.split("#", 1)
        if ent in ENTITY_SET and attr in ATTRIBUTE_SET:
            return f"{ent}#{attr}"

    # Domain-safe fallback
    return f"{ENTITY_SET[0]}#MISCELLANEOUS"


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

        for line in tqdm(fin, total=total_lines, desc="Extracting Quadruplets"):
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

            quads = extract_quads_from_llm_output(gen_text)

            formatted_quads = [
                {
                    "Aspect": a.strip(),
                    "Category": normalize_category(c),
                    "Opinion": o.strip(),
                    "VA": "0#0"
                }
                for a, c, o in quads
            ]

            fout.write(json.dumps({
                "ID": sample_id,
                "Quadruplet": formatted_quads
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

    process_jsonl(args.model, config.PREDICT_FILE, config.PREDICTION_FILE)


if __name__ == "__main__":
    main()
