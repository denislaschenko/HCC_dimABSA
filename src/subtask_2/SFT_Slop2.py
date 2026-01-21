import sys
import os
# Get the directory where this script is located
current_path = os.path.dirname(os.path.abspath(__file__))

# Go up two levels to reach the project root (src/subtask_2 -> src -> Root)
project_root = os.path.abspath(os.path.join(current_path, "../.."))

# Add the root to the system path so Python can find 'src'
sys.path.insert(0, project_root)
import json
import re
import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig  # <--- Add SFTConfig here
from transformers import EarlyStoppingCallback
from unsloth import FastLanguageModel
from tqdm import tqdm

# Import shared configuration
from src.shared import config

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
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    args = parser.parse_args()

    # --- Config Handling ---
    # We explicitly target 'subtask_2' for Task 2 (A, O, VA)
    # This ensures paths are correct even if global ACTIVE_SUBTASK is 'subtask_1'
    SUBTASK_NAME = "subtask_2"
    TASK_NAME = "task2"
    
    if SUBTASK_NAME not in config.TASK_CONFIGS:
        raise ValueError(f"{SUBTASK_NAME} not found in config TASK_CONFIGS")

    subtask_cfg = config.TASK_CONFIGS[SUBTASK_NAME]
    
    print(f"Running for: {SUBTASK_NAME} (Domain: {config.DOMAIN}, Lang: {config.LANG})")

    # --- Path Construction (Based on config.py logic) ---
    # 1. Data Directory: task-dataset/track_a/subtask_2/eng
    DATA_DIR = os.path.join(
        config.PROJECT_ROOT, 
        "task-dataset", 
        "track_a", 
        SUBTASK_NAME, 
        config.LANG
    )
    
    # 2. Output Directories
    MODEL_SAVE_DIR = os.path.join(config.PROJECT_ROOT, "outputs", SUBTASK_NAME, "models")
    PREDICTION_DIR = os.path.join(config.PROJECT_ROOT, "outputs", SUBTASK_NAME, "predictions")
    
    # 3. File Paths
    TRAIN_FILE = os.path.join(DATA_DIR, f"{config.LANG}_{config.DOMAIN}_train_alltasks.jsonl")
    PREDICT_FILE = os.path.join(DATA_DIR, f"{config.LANG}_{config.DOMAIN}_dev_{TASK_NAME}.jsonl")
    
    # Verify files exist
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    if not os.path.exists(PREDICT_FILE):
        # Fallback for test file if dev file missing
        print(f"Dev file not found at {PREDICT_FILE}. Checking for test file...")
        TEST_FILE = os.path.join(DATA_DIR, f"{config.LANG}_{config.DOMAIN}_test_{TASK_NAME}.jsonl")
        if os.path.exists(TEST_FILE):
            PREDICT_FILE = TEST_FILE
            print(f"Using test file: {PREDICT_FILE}")
        else:
            raise FileNotFoundError(f"Neither dev nor test file found in {DATA_DIR}")

    print(f"Data Dir: {DATA_DIR}")
    print(f"Train File: {TRAIN_FILE}")
    print(f"Predict File: {PREDICT_FILE}")

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
    train_raw = load_dataset("json", data_files=TRAIN_FILE, split="train")

    # Create a validation split (10%)
    split_dataset = train_raw.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"].map(format_train_example)
    eval_dataset = split_dataset["test"].map(format_train_example)

    # --- Trainer ---
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Use SFTConfig instead of TrainingArguments
    args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=config.EPOCHS if config.EPOCHS < 5 else 3,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        output_dir=MODEL_SAVE_DIR,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="none",
        dataset_text_field="text",  # <--- MOVED HERE
        # packing=True, # Optional: use this if you want better efficiency with short texts
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        # max_seq_length is controlled by the model/tokenizer
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        args=args, # Pass the SFTConfig here
        callbacks=[EarlyStoppingCallback(early_stopping_patience=subtask_cfg.get("PATIENCE", 2))],
    )

    print("\nTraining AO SFT model...")
    trainer.train()

    # --- Inference ---
    print("\nRunning inference...")
    FastLanguageModel.for_inference(model)

    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    # Set Output File name
    output_file = os.path.join(PREDICTION_DIR, f"pred_{config.LANG}_{config.DOMAIN}.jsonl")

    with open(PREDICT_FILE, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Predicting"):
            item = json.loads(line)
            prompt = build_infer_prompt(item["Text"])

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False, 
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Use Regex extraction
            triplets = extract_triplets_regex(decoded)

            fout.write(json.dumps({
                "ID": item["ID"],
                "Text": item["Text"],
                "Triplet": triplets
            }, ensure_ascii=False) + "\n")

    print(f"\nFinished. Output written to:\n{output_file}")
    from src.subtask_1.train_subtask1 import main as train_reg
    train_reg(output_file)


if __name__ == "__main__":
    main()
