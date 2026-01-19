import json
import argparse
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.subtask_1.train_subtask1 import main as train_reg
from src.shared import config


# =========================
# Config
# =========================
INPUT_FILE = config.LOCAL_PREDICT_FILE
OUTPUT_FILE = config.PREDICTION_FILE


# =========================
# SFT Instruction Prompt
# =========================
BASE_INSTRUCTION = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, O) tuples, where:
- A is an Aspect term (a phrase describing an entity mentioned in "Text")
- O is an Opinion term
- If the text states a fact that implies sentiment (e.g., "it is fast", "no features"), treat the factual descriptor as the Opinion.
- You must always format the output as a valid JSON object.
"""


def build_prompt(text: str):
    """
    Build SFT-style prompt
    """
    input_json = json.dumps(
        {"ID": "test_id", "Text": text},
        ensure_ascii=False
    )

    prompt = (
        BASE_INSTRUCTION
        + "\n### Question:\n"
        + "Now complete the following example. "
        + "Always output JSON in the same format and always predict \"VA\" as \"0#0\".\n"
        + f"Input:\n{input_json}\n\nOutput:"
    )

    return [
        {"role": "user", "content": prompt}
    ]


# =========================
# Output Parsing
# =========================
def extract_pairs_from_llm_output(output_text: str):
    """
    Robustly extract (Aspect, Opinion) pairs from model output.
    """
    try:
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        if start != -1 and end != -1:
            data = json.loads(output_text[start:end])
            triplets = data.get("Triplet", [])
            if isinstance(triplets, list):
                return [
                    (t["Aspect"], t["Opinion"])
                    for t in triplets
                    if isinstance(t, dict)
                    and "Aspect" in t
                    and "Opinion" in t
                ]
    except Exception:
        pass

    # Fallback regex (last resort)
    pattern = r'"Aspect"\s*:\s*"([^"]+)"\s*,\s*"Opinion"\s*:\s*"([^"]+)"'
    return re.findall(pattern, output_text)



# process json: Aspect Opinion V#A should be 0

def process_jsonl(model_name, input_path, output_path):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    model.eval()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Extracting (A, O)"):
            if not line.strip():
                continue

            item = json.loads(line)
            text = item["Text"]

            messages = build_prompt(text)

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            llm_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            pairs = extract_pairs_from_llm_output(llm_output)

            formatted_triplets = [
                {
                    "Aspect": a.strip(),
                    "Opinion": o.strip(),
                    "VA": "0#0"
                }
                for a, o in pairs
            ]

            fout.write(json.dumps({
                "ID": item["ID"],
                "Text": text,
                "Triplet": formatted_triplets
            }, ensure_ascii=False) + "\n")

    print("Extraction finished.")
    print("Passing output to VA regression script...")

    train_reg(output_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    )
    args = parser.parse_args()

    process_jsonl(args.model, INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
