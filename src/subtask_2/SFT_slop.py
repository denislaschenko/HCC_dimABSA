import json
import re
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel
from tqdm import tqdm

# Import shared configuration
from src.shared import config

# ==========================================
# 1. Model & Prompt Configuration
# ==========================================

# Since config.MODEL_NAME is set to BERT/RoBERTa models for regression,
# we define the LLM model ID specifically for this generative task.
# Ensure you have access to this model (HuggingFace login if private/requires access).
LLM_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

# Determine Task and Domain from Config
TASK_TYPE = config.TASK       # e.g., "task2" or "task3"
DOMAIN = config.DOMAIN        # e.g., "restaurant", "laptop"
LANG = config.LANG            # e.g., "eng"

def get_prompt_templates(task, domain):
    """Returns the instruction string and label definitions based on task and domain."""
    
    # Define Labels Map for Task 3
    entity_attribute_map = {
        'restaurant': (
            'RESTAURANT, FOOD, DRINKS, AMBIENCE, SERVICE, LOCATION',
            'GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS'
        ),
        'laptop': (
            'LAPTOP, DISPLAY, KEYBOARD, MOUSE, MOTHERBOARD, CPU, FANS_COOLING, PORTS, MEMORY, POWER_SUPPLY, OPTICAL_DRIVES, BATTERY, GRAPHICS, HARD_DISK, MULTIMEDIA_DEVICES, HARDWARE, SOFTWARE, OS, WARRANTY, SHIPPING, SUPPORT, COMPANY',
            'GENERAL, PRICE, QUALITY, DESIGN_FEATURES, OPERATION_PERFORMANCE, USABILITY, PORTABILITY, CONNECTIVITY, MISCELLANEOUS'
        ),
        'hotel': (
            'HOTEL, ROOMS, FACILITIES, ROOM_AMENITIES, SERVICE, LOCATION, FOOD_DRINKS',
            'GENERAL, PRICE, COMFORT, CLEANLINESS, QUALITY, DESIGN_FEATURES, STYLE_OPTIONS, MISCELLANEOUS'
        ),
        'finance': (
            'MARKET, COMPANY, BUSINESS, PRODUCT',
            'GENERAL, SALES, PROFIT, AMOUNT, PRICE, COST'
        ),
    }

    base_instruction = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all {triplet_desc}, where:
{triplet_defs}

Valence ranges from 1 (negative) to 9 (positive),
Arousal ranges from 1 (calm) to 9 (excited).
"""

    if task == "task2":
        triplet_desc = "(A, O, VA) triplets"
        triplet_defs = """- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)"""

        example_block = """### Example:
Input:
[Text] average to good thai food, but terrible delivery.

Output:
[Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)"""

    elif task == "task3":
        triplet_desc = "(A, C, O, VA) quadruplets"
        triplet_defs = """- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- C is a Category label (e.g. FOOD#QUALITY)
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)"""

        # Handle domain lookup safely
        labels = entity_attribute_map.get(domain, ("", ""))
        ent_labels, attr_labels = labels
        
        constraints = f"""
### Label constraints:
[Entity Labels] ({ent_labels})
[Attribute Labels] ({attr_labels})"""

        example_block = f"""{constraints}

### Example:
Input:
[Text] average to good thai food, but terrible delivery.

Output:
[Quadruplet] (thai food, FOOD#QUALITY, average to good, 6.75#6.38),
             (delivery, SERVICE#GENERAL, terrible, 2.88#6.62)"""
    else:
        # Fallback for Task 1 (if run accidentally), though this script is optimized for 2/3
        triplet_desc = "(A, O) tuples"
        triplet_defs = """- A is an Aspect term
- O is an Opinion term"""
        example_block = ""

    instruction = f"{base_instruction}\n{example_block}\n\n### Question:\nNow complete the following example:\nInput:"
    return instruction

# ==========================================
# 2. Data Formatting & Parsing
# ==========================================

def format_train_example(example, task):
    """Formats data into SFT text using the Tuple approach."""
    text = example["Text"]
    quads = example.get("Quadruplet", [])
    
    # Build Answer String
    answer_parts = []
    for q in quads:
        if q.get("Aspect") == "NULL" or q.get("Opinion") == "NULL":
            continue
            
        if task == "task1":
            # Task 1 usually doesn't have VA, but just in case config points here
            answer_parts.append(f"({q['Aspect']}, {q['Opinion']})")
        elif task == "task2":
            answer_parts.append(f"({q['Aspect']}, {q['Opinion']}, {q['VA']})")
        elif task == "task3":
            answer_parts.append(f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})")
            
    answer_str = ", ".join(answer_parts)
    
    # Construct Prompt
    prompt = f"{example['instruction']}\n[Text] {text}\n\nOutput:\n"
    return {"text": f"{prompt}\n{answer_str}"}

def build_infer_prompt(instruction, text):
    """Builds prompt for inference, stopping before the answer."""
    return f"{instruction}\n[Text] {text}\n\nOutput:\n"

def extract_predictions(text, task):
    """Robust Regex extraction of tuples."""
    result = []
    
    if task == "task1":
        pattern = r'\(([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, text)
        for aspect, opinion in matches:
            result.append({
                "Aspect": aspect.strip(),
                "Opinion": opinion.strip(),
                "VA": "0#0" # Dummy for task 1 if needed
            })
            
    elif task == "task2":
        # Pattern: (Aspect, Opinion, VA#VA)
        pattern = r'\(([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
        matches = re.findall(pattern, text)
        
        for aspect, opinion, va in matches:
            result.append({
                "Aspect": aspect.strip(),
                "Opinion": opinion.strip(),
                "VA": va.strip()
            })

    elif task == "task3":
        # Pattern: (Aspect, Category#Sub, Opinion, VA#VA)
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
        matches = re.findall(pattern, text)
        
        for aspect, category, opinion, va in matches:
            result.append({
                "Aspect": aspect.strip(),
                "Category": category.strip(),
                "Opinion": opinion.strip(),
                "VA": va.strip()
            })
            
    return result

# ==========================================
# 3. Main Execution
# ==========================================

def main():
    print(f"Starting Training for {config.ACTIVE_SUBTASK} ({TASK_TYPE}) on {DOMAIN}...")

    # --- Prepare Prompts ---
    instruction = get_prompt_templates(TASK_TYPE, DOMAIN)

    # --- Load Model (Unsloth) ---
    print(f"Loading model: {LLM_MODEL_ID}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LLM_MODEL_ID,
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

    # --- Prepare Data ---
    print(f"Loading training data from: {config.LOCAL_TRAIN_FILE}")
    
    # Check if local training file exists, otherwise fall back to standard train file
    train_path = config.LOCAL_TRAIN_FILE
    if not os.path.exists(train_path):
        print(f"Warning: {train_path} not found. Falling back to {config.TRAIN_FILE}")
        train_path = config.TRAIN_FILE
        
    raw_dataset = load_dataset("json", data_files=train_path, split="train")
    
    # 1. Inject instruction into dataset for easy access
    raw_dataset = raw_dataset.map(lambda x: {"instruction": instruction})
    
    # 2. Split for Validation
    split_dataset = raw_dataset.train_test_split(test_size=0.1)
    
    # 3. Format prompts
    train_dataset = split_dataset["train"].map(
        lambda x: format_train_example(x, TASK_TYPE),
        remove_columns=raw_dataset.column_names
    )
    eval_dataset = split_dataset["test"].map(
        lambda x: format_train_example(x, TASK_TYPE),
        remove_columns=raw_dataset.column_names
    )

    # --- Trainer ---
    # Ensure output directory exists
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=1, 
            gradient_accumulation_steps=4,
            warmup_steps=20,
            num_train_epochs=config.EPOCHS if config.EPOCHS < 5 else 3, # Cap epochs for LLM SFT to 3 usually
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=config.SEED,
            output_dir=config.MODEL_SAVE_DIR,
            report_to="none",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)],
    )

    print("Starting training...")
    trainer.train()

    # --- Inference ---
    print("Running inference...")
    FastLanguageModel.for_inference(model)
    
    # Ensure prediction directory exists
    os.makedirs(config.PREDICTION_DIR, exist_ok=True)
    
    predict_dataset = load_dataset("json", data_files=config.LOCAL_PREDICT_FILE, split="train")
    
    output_key = "Triplet" if TASK_TYPE in ["task1", "task2"] else "Quadruplet"
    
    with open(config.PREDICTION_FILE, "w", encoding="utf-8") as fout:
        for item in tqdm(predict_dataset):
            prompt_text = build_infer_prompt(instruction, item["Text"])
            
            inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256, 
                use_cache=True,
                temperature=None, 
                do_sample=False,  
                pad_token_id=tokenizer.eos_token_id
            )
            
            decoded_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            extracted_data = extract_predictions(decoded_output, TASK_TYPE)
            
            result = {
                "ID": item.get("ID", ""),
                "Text": item["Text"],
                output_key: extracted_data
            }
            
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Finished. Predictions saved to {config.PREDICTION_FILE}")
    
    # If you have a subsequent regression step (like for Subtask 1), uncomment:
    from src.subtask_1.train_subtask1 import main as train_reg
    train_reg(config.PREDICTION_FILE)

if __name__ == "__main__":
    main()
