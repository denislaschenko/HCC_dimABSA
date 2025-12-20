import json
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from src.subtask_1.train_subtask1 import main as train_reg
from src.shared import config


ICL_PROMPT = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, O) tuples, where:
- A is an Aspect term (a phrase describing an entity mentioned in "Text")
- O is an Opinion term

### Example:
Input:
{"ID": "lap26_aspect_va_dev_30", "Text": "Plenty of memory and processing power but way too many programs and apps"}

Output:
{"ID": "laptop_quad_dev_1", "Text": "this unit is pretty and stylish , so my high school daughter was attracted to it for that reason .", "Triplet": [{"Aspect": "unit", "Opinion": "pretty", "VA": "0#0"}, {"Aspect": "unit", "Opinion": "stylish", "VA": "0#0"}]}

### Question:
Now complete the following example, never change the Layout described in the Output examples and always predict "VA" to be "0#0":
Input:
"""

input_file = config.PREDICT_FILE
output_file = config.PREDICTION_FILE

def build_prompt(text: str):
    """Create full ICL prompt for a given text."""
    prompt = (
            ICL_PROMPT +
            f"[Text] {text}\n\nOutput:"
    )
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def extract_pairs_from_llm_output(output_text: str):
    pattern = r'"Aspect":\s*"([^"]+)"\s*,\s*"Opinion":\s*"([^"]+)"'
    matches = re.findall(pattern, output_text)
    return matches


def process_jsonl(model_name, input_path, output_path):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    model.eval()

    print(f"Reading input: {input_path}")

    try:
        total_lines = sum(1 for _ in open(input_path, 'r', encoding="utf-8"))
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    print(f"Writing output: {output_path}")

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, total=total_lines, desc="Extracting"):
            if not line.strip(): continue

            item = json.loads(line)
            text = item["Text"]

            prompt = build_prompt(text)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs.input_ids.shape[1]
            generated_tokens = generated[0][input_len:]
            llm_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            pairs = extract_pairs_from_llm_output(llm_output)

            formatted_triplets = []
            for aspect, opinion in pairs:
                formatted_triplets.append({
                    "Aspect": aspect.strip(),
                    "Opinion": opinion.strip(),
                    "VA": "0#0"
                })



            new_entry = {
                "ID": item["ID"],
                "Text": text,
                "Triplet": formatted_triplets
            }

            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            fout.flush()

    print("Done.")

    train_reg(output_file)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")

    args = parser.parse_args()
    process_jsonl(args.model, input_file, output_file)


if __name__ == "__main__":
    main()