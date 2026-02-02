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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
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
    # Using a standard Qwen model
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    # --- Config Handling ---
    SUBTASK_NAME = "subtask_2"
    TASK_NAME = "task2"
    
    if SUBTASK_NAME not in config.TASK_CONFIGS:
        raise ValueError(f"{SUBTASK_NAME} not found in config TASK_CONFIGS")

    subtask_cfg = config.TASK_CONFIGS[SUBTASK_NAME]
    
    print(f"Running for: {SUBTASK_NAME} (Domain: {config.DOMAIN}, Lang: {config.LANG})")

    # --- Path Construction ---
    DATA_DIR = os.path.join(
        config.PROJECT_ROOT, 
        "task-dataset", 
        "track_a", 
        SUBTASK_NAME, 
        config.LANG
    )
    
    MODEL_SAVE_DIR = os.path.join(config.PROJECT_ROOT, "outputs", SUBTASK_NAME, "models")
    PREDICTION_DIR = os.path.join(config.PROJECT_ROOT, "outputs", SUBTASK_NAME, "predictions")
    
    TRAIN_FILE = os.path.join(DATA_DIR, f"{config.LANG}_{config.DOMAIN}_train_alltasks.jsonl")
    PREDICT_FILE = os.path.join(DATA_DIR, f"{config.LANG}_{config.DOMAIN}_dev_{TASK_NAME}.jsonl")
    
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    if not os.path.exists(PREDICT_FILE):
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

    # --- Load Model (Standard HuggingFace) ---
    print(f"Loading model: {args.model}...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model with Quantization (Optional but recommended for VRAM)
    # Note: BitsAndBytes works on Windows with supported CUDA drivers
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Disable cache for training
    model.config.use_cache = False

    # --- LoRA Config (Standard PEFT) ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load & Split Data ---
    train_raw = load_dataset("json", data_files=TRAIN_FILE, split="train")

    # Create a validation split (10%)
    split_dataset = train_raw.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"].map(format_train_example)
    eval_dataset = split_dataset["test"].map(format_train_example)

    # --- Trainer ---
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Reverting to standard TrainingArguments for maximum compatibility
    training_args = TrainingArguments(
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
        fp16=True, # Standard mixed precision
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        dataset_text_field="text",
        # Standard TRL usually requires tokenizer, not processing_class
        tokenizer=tokenizer,
        max_seq_length=2048,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=subtask_cfg.get("PATIENCE", 2))],
    )

    # --- Start Training ---
    print("\nTraining AO SFT model...")
    trainer.train()

    # --- Inference ---
    print("\nRunning inference...")
    model.eval() # Standard eval mode

    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
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
    
    # Regression step
    from legacy.train_subtask1 import main as train_reg
    train_reg(output_file)

if __name__ == "__main__":
    main()
