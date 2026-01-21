import json
import re
import os
import argparse
import torch
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel
from tqdm import tqdm

# Assuming these files exist in your project structure
from src.shared import config
# We keep the import for the regression step, though it might only be relevant for Task 1
from src.subtask_1.train_subtask1 import main as train_reg

# ==========================================
# 1. Prompt Configuration (Task 2 - Tuple Approach)
# ==========================================

INSTRUCTION = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

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

def format_train_example(example):
    """
    Converts training jsonl → SFT text (Tuple Format)
    """
    text = example["Text"]
    quads = example.get("Quadruplet", [])
    
    # Build Answer String (A, O, VA)
    answer_parts = []
    for q in quads:
        if q.get("Aspect") != "NULL" and q.get("Opinion") != "NULL":
            answer_parts.append(f"({q['Aspect']}, {q['Opinion']}, {q['VA']})")
            
    answer_str = ", ".join(answer_parts)
    
    # Construct Prompt
    prompt = f"{INSTRUCTION}\n[Text] {text}\n\nOutput:\n"
    return {"text": f"{prompt}\n{answer_str}"}

def build_infer_prompt(text):
    return f"{INSTRUCTION}\n[Text] {text}\n\nOutput:\n"

def extract_triplets_regex(decoded_text):
    """Robust Regex extraction of (A, O, VA) tuples."""
    result = []
    # Pattern: (Aspect, Opinion, VA#VA)
    # [^,]+ matches any character except comma (Aspect/Opinion)
    # [\d.]+#[\d.]+ matches the score
    pattern = r'\(([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
    matches = re.findall(pattern, decoded_text)
    
    for aspect, opinion, va in matches:
        result.append({
            "Aspect": aspect.strip(),
            "Opinion": opinion.strip(),
            "VA": va.strip()
        })
    return result

# ==========================================
# 2. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    # Defaulting to a Qwen model compatible with Unsloth
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    args = parser.parse_args()

    # --- Fix Pathing ---
    # Force subtask_2 paths to avoid config mismatches if ACTIVE_SUBTASK is set to subtask_1
    SUBTASK_DIR = "subtask_2"
    DATA_BASE_DIR = os.path.join(config.PROJECT_ROOT, "task-dataset", "track_a", SUBTASK_DIR, config.LANG)
    
    # Define explicit paths
    local_train_path = os.path.join(DATA_BASE_DIR, "local", "local_train.jsonl")
    standard_train_path = os.path.join(DATA_BASE_DIR, f"{config.LANG}_{config.DOMAIN}_train_alltasks.jsonl")
    
    # Select training path
    train_path = local_train_path if os.path.exists(local_train_path) else standard_train_path
    
    # Define inference path
    # Assuming input for inference is in the local folder or defined in config
    infer_path = os.path.join(DATA_BASE_DIR, "local", "local_dev_input.jsonl")
    if not os.path.exists(infer_path):
        infer_path = config.PREDICT_FILE

    print(f"Using Training Data: {train_path}")
    print(f"Using Inference Data: {infer_path}")

    # --- Load Model (Unsloth) ---
    print(f"Loading model: {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048, 
        load_in_4bit=True,
    )

    # --- LoRA Config (Unsloth) ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=config.SEED,
    )

    # --- Load & Split Data ---
    train_raw = load_dataset(
        "json",
        data_files=train_path, 
        split="train"
    )

    # Create a validation split
    split_dataset = train_raw.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"].map(format_train_example)
    eval_dataset = split_dataset["test"].map(format_train_example)

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Pass eval_dataset here
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=config.EPOCHS if config.EPOCHS < 5 else 3,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            output_dir=config.MODEL_SAVE_DIR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            report_to="none",
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)],
    )

    print("\nTraining AO SFT model...")
    trainer.train()

    # --- Inference ---
    print("\nRunning inference...")
    FastLanguageModel.for_inference(model)

    # Ensure output directory exists
    os.makedirs(config.PREDICTION_DIR, exist_ok=True)
    
    # Set Output File
    output_file = os.path.join(config.PREDICTION_DIR, f"pred_{config.LANG}_{config.DOMAIN}.jsonl")

    with open(infer_path, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Predicting"):
            item = json.loads(line)
            prompt = build_infer_prompt(item["Text"])

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False, # Deterministic decoding
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Use Regex extraction instead of JSON parsing
            triplets = extract_triplets_regex(decoded)

            fout.write(json.dumps({
                "ID": item["ID"],
                "Text": item["Text"],
                "Triplet": triplets
            }, ensure_ascii=False) + "\n")

    print(f"\nFinished. Output written to:\n{output_file}")
    
    # If you want to run the regression step for Task 1 logic (optional for Task 2)
    # train_reg(output_file)

if __name__ == "__main__":
    main()
