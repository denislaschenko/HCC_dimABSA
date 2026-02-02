import sys
import os
import time
import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared import config
import src.subtask_2.inference_clean as inference

INSTRUCTION = """Given a textual instance [Text], extract all (A, O, VA) triplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

RULES:
1. EXTRACT EXACTLY: Copy the substring exactly as it appears in the text, including typos and capitalization.
2. NO NULLS: Every triplet must have explicit Aspects and Opinions from the text. Do not generate 'NULL'.
3. NO HALLUCINATIONS: If a word is not in the text, do not extract it.
4. VA FORMAT: Predict the specific Valence#Arousal score (e.g., 7.5#4.2).

Be exhaustive: extract every aspect and opinion mentioned. You must preserve the EXACT capitalization and spelling as it appears in the [Text]."""


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    args_cli = parser.parse_args()

    print(f"Training SFT Model for Subtask 2 ({config.LANG}/{config.DOMAIN})")
    print(f"Training Data: {config.TRAIN_FILE}")
    print(f"Inference Input: {config.PREDICT_FILE}")
    print(f"Inference Output: {config.OUTPUT_FILE}")

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(args_cli.base_model, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args_cli.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    def format_train(ex):
        valid_triplets = [
            f"({q['Aspect']}, {q['Opinion']}, {q['VA']})"
            for q in ex.get("Quadruplet", [])
            if q.get("Aspect") != "NULL"
        ]
        ans = ", ".join(valid_triplets)

        msg = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"[Text] {ex['Text']}\n\nOutput:\n"},
            {"role": "assistant", "content": ans}
        ]
        return {"text": tokenizer.apply_chat_template(msg, tokenize=False)}

    train_ds = load_dataset("json", data_files=config.TRAIN_FILE, split="train").train_test_split(test_size=0.1, seed=config.SEED)
    train_set = train_ds["train"].map(format_train)
    eval_set = train_ds["test"].map(format_train)

    sft_args = SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=3,
        output_dir=config.MODEL_SAVE_DIR, # Using Config
        bf16=True,
        dataset_text_field="text",
        packing=False,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=10
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("\n--- Starting Training ---")
    trainer.train()

    final_adapter_path = os.path.join(config.MODEL_SAVE_DIR, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    print(f"Adapter saved to {final_adapter_path}")

    print("\n--- Running Inference ---")
    del model, trainer
    torch.cuda.empty_cache()

    inference.run_inference(
        checkpoint_path=final_adapter_path,
        input_file=config.PREDICT_FILE,
        output_file=config.OUTPUT_FILE,
        base_model_id=args_cli.base_model
    )

    end_time = time.time()
    print(f"\nTotal Execution Time: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()