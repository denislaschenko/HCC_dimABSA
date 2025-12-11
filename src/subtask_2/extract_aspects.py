import json
import argparse
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm  # 🟢 Added for progress visibility

# ---------------------------
# ICL Template (Task 2)
# ---------------------------
ICL_PROMPT = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, O) tupels, where:
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

def build_prompt(text: str):
    """Create full ICL prompt for a given text."""
    prompt = (
        ICL_PROMPT +
        f"[Text] {text}\n\nOutput:"
    )
    # Qwen/Llama specific chat formatting
    return f"<|user|>\n{prompt}\n<|assistant|>\n"

def extract_pairs_from_llm_output(output_text: str):
    pattern = r'"Aspect":\s*"([^"]+)"\s*,\s*"Opinion":\s*"([^"]+)"'
    matches = re.findall(pattern, output_text)
    return matches

def process_jsonl(model_name, input_path, output_path):
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 🟢 FIX: Use device_map="auto" for 4-bit models. 
    # Do NOT use .cuda() manually.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype="auto"
    )
    model.eval()

    print(f"Reading input: {input_path}")
    
    # 🟢 FIX: Count lines first to setup progress bar
    total_lines = sum(1 for _ in open(input_path, 'r', encoding="utf-8"))
    
    print(f"Writing output: {output_path}")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        # 🟢 FIX: Wrap in tqdm for progress bar
        for line in tqdm(fin, total=total_lines, desc="Extracting"):
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
                    pad_token_id=tokenizer.eos_token_id # 🟢 FIX: Explicitly set pad token
                )

            input_len = inputs.input_ids.shape[1]
            generated_tokens = generated[0][input_len:]
            llm_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            pairs = extract_pairs_from_llm_output(llm_output)


            aspects = [p[0] for p in pairs]
            opinions = [p[1] for p in pairs]

            new_entry = {
                "ID": item["ID"],
                "Text": text,
                # Speichert es als Liste von Listen: [["display", "bright"], ["color gamut", "wide"]]
                "Triplets": pairs
            }

            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            fout.flush() # 🟢 FIX: Ensure data is written to disk immediately

            print(llm_output)

    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit") 
    # Note: Checked model name, assuming you meant Qwen2.5 or similar. 
    # If "Qwen3" is a private model, keep your original string.

    args = parser.parse_args()
    process_jsonl(args.model, args.input, args.output)

if __name__ == "__main__":
    main()

