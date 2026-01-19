import json
import argparse
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from src.shared import config



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

    prompt = (
        f"{INSTRUCTION}\n"
        f"Input:\n{json.dumps({'ID': example['ID'], 'Text': example['Text']})}\n\n"
        f"Output:\n{json.dumps({'ID': example['ID'], 'Text': example['Text'], 'Triplet': triplets})}"
    )

    return {"text": prompt}


def build_infer_prompt(text, idx):
    return (
        f"{INSTRUCTION}\n"
        f"Input:\n{json.dumps({'ID': idx, 'Text': text})}\n\n"
        f"Output:"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
    args = parser.parse_args()

   
    # Load tokenizer + model
 
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16
    )

    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

   
    # Load & format training data

    train_raw = load_dataset(
        "json",
        data_files=config.LOCAL_TRAIN_FILE,
        split="train"
    )

    train_dataset = train_raw.map(format_train_example)
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
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
            output_dir=config.MODEL_SAVE_DIR,
            fp16=True,
            report_to="none",
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)],
    )

    print("\nTraining AO SFT model...")
    trainer.train()

   
    # Inference
    
    print("\n Running inference...")
    model.eval()

    with open(config.LOCAL_PREDICT_FILE, "r", encoding="utf-8") as fin, \
         open(config.PREDICTION_FILE, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Predicting"):
            item = json.loads(line)
            prompt = build_infer_prompt(item["Text"], item["ID"])

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
                pred = json.loads(decoded[start:end])
                triplets = pred.get("Triplet", [])
            except Exception:
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

