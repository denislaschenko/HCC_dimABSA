
"""
Extracts Aspect terms using the same ICL prompt template
used in DimASTE subtask_2 training.

Input:  JSONL with fields {ID, Text}
Output: JSONL with fields {ID, Text, Aspect: [...]}
"""

import json
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# ---------------------------
# ICL Template (Task 2)
# ---------------------------
ICL_PROMPT = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

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
"""


def build_prompt(text: str):
    """Create full ICL prompt for a given text."""
    prompt = (
        ICL_PROMPT +
        f"[Text] {text}\n\nOutput:"
    )
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


# ---------------------------
# Extract Triplet → Aspect
# ---------------------------
def extract_aspects_from_llm_output(output_text: str):
    """
    Extracts Aspect names from an LLM-generated triplet output.

    Example LLM output:
    [Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)
    """
    pattern = r'\(([^,]+),\s*([^,]+),\s*[\d.]+#[\d.]+\)'
    matches = re.findall(pattern, output_text)

    aspects = [m[0].strip() for m in matches]
    return aspects


# ---------------------------
# Main processing function
# ---------------------------
def process_jsonl(model_name, input_path, output_path):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    model.eval()

    print(f"Reading input: {input_path}")
    print(f"Writing output: {output_path}")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)
            text = item["Text"]

            # Build ICL prompt
            prompt = build_prompt(text)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.3,
                    top_p=0.9,
                )

            decoded = tokenizer.decode(generated[0])
            llm_output = decoded.split("<|assistant|>")[-1].strip()

            # Extract aspects from LLM output
            aspects = extract_aspects_from_llm_output(llm_output)

            # Write new JSON line
            new_entry = {
                "ID": item["ID"],
                "Text": text,
                "Aspect": aspects
            }
            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

    print("Done.")


# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract aspects using DimASTE Task2 ICL prompting."
    )

    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
        help="LLM model to use for ICL extraction."
    )

    args = parser.parse_args()
    process_jsonl(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
