import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Assuming these files exist in your project structure
from src.shared import config
from src.subtask_1.train_subtask1 import main as train_reg

INSTRUCTION = """Below is an instruction describing a task, paired with an input that provides additional context.

### Instruction:
Given a textual instance [Text], extract all (A, O) tuples, where:
- A is an Aspect term
- O is an Opinion term
- Output must be valid JSON
"""

def format_train_example(example):
    """
    Converts training jsonl → SFT text
    """
    triplets = []
    for q in example.get("Quadruplet", []):
        if q.get("Aspect") != "NULL" and q.get("Opinion") != "NULL":
            triplets.append({
                "Aspect": q["Aspect"],
                "Opinion": q["Opinion"]
            })

    # Use ensure_ascii=False to keep original characters (e.g., Chinese) visible to the model
    input_json = json.dumps({'ID': example['ID'], 'Text': example['Text']}, ensure_ascii=False)
    output_json = json.dumps({
        'ID': example['ID'], 
        'Text': example['Text'], 
        'Triplet': triplets
    }, ensure_ascii=False)

    prompt = (
        f"{INSTRUCTION}\n"
        f"Input:\n{input_json}\n\n"
        f"Output:\n{output_json}"
    )

    return {"text": prompt}


def build_infer_prompt(text, idx):
    input_json = json.dumps({'ID': idx, 'Text': text}, ensure_ascii=False)
    return (
        f"{INSTRUCTION}\n"
        f"Input:\n{input_json}\n\n"
        f"Output:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
    args = parser.parse_args()

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model with Quantization Config (Mistake Fix)
    # Explicitly define 4-bit quantization config to ensure correct loading
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

    # 3. Disable cache for training (Mistake Fix)
    # Required when using gradient checkpointing or peft to avoid warnings/errors
    model.config.use_cache = False

    # 4. LoRA Config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # 5. Load & Split Data (Mistake Fix)
    # Loading data
    train_raw = load_dataset(
        "json",
        data_files=config.TRAIN_FILE, 
        split="train"
    )

    # Create a validation split to support evaluation_strategy and EarlyStopping
    split_dataset = train_raw.train_test_split(test_size=0.1) # 10% for validation
    train_dataset = split_dataset["train"].map(format_train_example)
    eval_dataset = split_dataset["test"].map(format_train_example)

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Pass eval_dataset here
        dataset_text_field="text",
        max_seq_length=512,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=10,
            logging_steps=50,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False, # Loss should be minimized
            output_dir=config.MODEL_SAVE_DIR,
            fp16=True, # fp16 is generally redundant with load_in_4bit, but kept for compatibility
            report_to="none",
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)],
    )

    print("\nTraining AO SFT model...")
    trainer.train()

    # 7. Inference
    print("\nRunning inference...")
    model.eval()

    with open(config.PREDICT_FILE, "r", encoding="utf-8") as fin, \
         open(config.PREDICTION_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Predicting"):
            item = json.loads(line)
            prompt = build_infer_prompt(item["Text"], item["ID"])

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

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

            try:
                start = decoded.find("{")
                end = decoded.rfind("}") + 1
                
                if start != -1 and end > start:
                    json_str = decoded[start:end]
                    pred = json.loads(json_str)
                    triplets = pred.get("Triplet", [])
                else:
                    triplets = []
            except Exception as e:
                print(f"Error parsing JSON for ID {item['ID']}: {e}")
                triplets = []

            final_triplets = [
                {
                    "Aspect": t["Aspect"],
                    "Opinion": t["Opinion"],
                    "VA": "0#0"
                }
                for t in triplets
                if "Aspect" in t and "Opinion" in t
            ]

            fout.write(json.dumps({
                "ID": item["ID"],
                "Text": item["Text"],
                "Triplet": final_triplets
            }, ensure_ascii=False) + "\n")

    print(f"\nFinished. Output written to:\n{config.PREDICTION_FILE}")
    train_reg(config.PREDICTION_FILE)


if __name__ == "__main__":
    main()




