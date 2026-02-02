



import sys
import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# Path setup
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared import config


def main():
    parser = argparse.ArgumentParser()
+\
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the SFT checkpoint folder", default="outputs/subtask_2/models/checkpoint-900")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    # Paths
    DPO_DATA_FILE = r"/task-dataset/track_a/subtask_2/eng/dpo_pairs_laptop.jsonl"
    OUTPUT_DIR = "outputs/subtask_2/dpo_models"

    if not os.path.exists(DPO_DATA_FILE):
        print(f"Error: DPO data not found at {DPO_DATA_FILE}. Run create_dpo_data.py first.")
        return

    print(f"Loading Base Model: {args.base_model}...")

    # 1. Load Base Model (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,

    +    device_map={"": 0},
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the SFT Adapter
    print(f"Loading SFT Adapter from: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)

    # 3. Merge SFT Adapter into Base (Best practice for DPO stability)
    print("Merging SFT adapter into base model...")
    model = model.merge_and_unload()

    model.enable_input_require_grads()

    # 4. Create NEW LoRA Config for DPO
    # We train a fresh adapter specifically for preference alignment
    peft_config = LoraConfig(
        r=8,  # Smaller rank is usually fine for DPO
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Load Dataset
    dataset = load_dataset("json", data_files=DPO_DATA_FILE, split="train")
    # Split small validation set
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # 6. Configure DPO
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-7,  # Very low LR for DPO (Standard is 1e-6 to 5e-7)
        num_train_epochs=0.5,  # 1 Epoch is usually enough for DPO
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        beta=0.5,  # The temperature of DPO (0.1 is standard)
        max_length=1024,
        max_prompt_length=512,
        remove_unused_columns=False,
        report_to="none",
        fp16=True,
    )

    # 7. Initialize Trainer
    print("Starting DPO Training...")
    dpo_trainer = DPOTrainer(
        model=model,  # The "Policy" model (SFT-merged)
        ref_model=None,  # Trainer creates a copy automatically for Reference
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,  # Train a new adapter
    )

    # 8. Train
    dpo_trainer.train()

    # 9. Save
    final_save_path = os.path.join(OUTPUT_DIR, "final_dpo_adapter")
    dpo_trainer.save_model(final_save_path)
    print(f"DPO Training Complete. Saved to: {final_save_path}")


if __name__ == "__main__":
    main()