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
from transformers import EarlyStoppingCallback, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer, util

# Import shared configuration
from src.shared import config

# ==========================================
# 1. Prompt Configuration (Task 2 - Tuple Approach)
# ==========================================

INSTRUCTION = """Given a textual instance [Text], extract all (A, O, VA) triplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

RULES:
1. EXTRACT EXACTLY: Copy the substring exactly as it appears in the text, including typos and capitalization.
2. NO NULLS: Every triplet must have explicit Aspects and Opinions from the text. Do not generate 'NULL'.
3. NO HALLUCINATIONS: If a word is not in the text, do not extract it.
4. VA FORMAT: Use '0#0' as a placeholder for the VA score; it will be calculated by a secondary model.

Be exhaustive: extract every aspect and opinion mentioned. You must preserve the EXACT capitalization and spelling as it appears in the [Text]."""


class ExampleRetriever:
    def __init__(self, train_path, model_name='BAAI/bge-base-en-v1.5'):
        self.encoder = SentenceTransformer(model_name)
        self.examples = []

        # Load examples from the training file
        with open(train_path, 'r', encoding='UTF-8') as f:
            raw_data = [json.loads(line) for line in f if line.strip()]

        for item in raw_data:
            quads = item.get("Quadruplet", [])
            # Filter for valid triplets only
            valid_triplets = [f"({q['Aspect']}, {q['Opinion']}, {q['VA']})"
                              for q in quads if q.get("Aspect") != "NULL"]

            if valid_triplets:
                self.examples.append({
                    "Text": item["Text"],
                    "Answer": ", ".join(valid_triplets)
                })

        # Pre-compute embeddings for fast retrieval
        texts = [ex["Text"] for ex in self.examples]
        self.embeddings = self.encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def retrieve(self, query_text, k=3):
        query_emb = self.encoder.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_results = torch.topk(scores, k=min(k, len(self.examples)))
        return [self.examples[idx] for idx in top_results.indices]

# ==========================================
# 2. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    args_cli = parser.parse_args()

    # Paths and Config
    subtask_cfg = config.TASK_CONFIGS["subtask_2"]
    DATA_DIR = os.path.join(config.PROJECT_ROOT, "task-dataset", "track_a", "subtask_2", config.LANG)
    MODEL_SAVE_DIR = os.path.join(config.PROJECT_ROOT, "outputs", "subtask_2", "models")
    PREDICTION_DIR = os.path.join(config.PROJECT_ROOT, "outputs", "subtask_2", "predictions")
    TRAIN_FILE = os.path.join(DATA_DIR, f"{config.LANG}_{config.DOMAIN}_production_train.jsonl")
    PREDICT_FILE = os.path.join(DATA_DIR, f"{config.LANG}_{config.DOMAIN}_dev_task2.jsonl")

    # --- Load Model & Tokenizer ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(args_cli.model, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args_cli.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- LoRA Config ---
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # --- Training Setup ---
    def format_train(ex):
        # Change the VA scores to 0#0 placeholder in the training targets
        ans = ", ".join([f"({q['Aspect']}, {q['Opinion']}, 0#0)" for q in ex.get("Quadruplet", []) if
                         q.get("Aspect") != "NULL"])
        msg = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"[Text] {ex['Text']}\n\nOutput:\n"},
            {"role": "assistant", "content": ans}
        ]
        return {"text": tokenizer.apply_chat_template(msg, tokenize=False)}

    train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train").train_test_split(test_size=0.1)
    train_set = train_ds["train"].map(format_train)
    eval_set = train_ds["test"].map(format_train)

    sft_args = SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        num_train_epochs=3,
        output_dir=MODEL_SAVE_DIR,
        bf16=True,
        dataset_text_field="text",
        packing=False,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        args=sft_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    print("\nTraining AO SFT model...")
    trainer.train()

    final_adapter_path = os.path.join(MODEL_SAVE_DIR, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    print(f"Model saved to {final_adapter_path}")

    print("\n--- Starting Integrated Inference ---")
    import subprocess

    inference_script = os.path.join(config.PROJECT_ROOT, "src", "subtask_2", "inference.py")

    subprocess.run([
        sys.executable,
        inference_script,
        "--checkpoint", final_adapter_path
    ])

    # # --- RAG Inference ---
    # print("\nInitializing Retriever for Inference...")
    # retriever = ExampleRetriever(TRAIN_FILE)
    # output_file = os.path.join(PREDICTION_DIR, f"pred_{config.LANG}_{config.DOMAIN}.jsonl")
    # os.makedirs(PREDICTION_DIR, exist_ok=True)
    #
    # def build_infer_prompt(text, retrieved_examples):
    #     messages = [{"role": "system", "content": INSTRUCTION}]
    #     for ex in retrieved_examples:
    #         messages.append({"role": "user", "content": f"Extract triplets: {ex['Text']}"})
    #         messages.append({"role": "assistant", "content": ex["Answer"]})
    #     messages.append({"role": "user", "content": f"[Text] {text}\n\nOutput:\n"})
    #     return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #
    # def extract_triplets_regex(decoded_text):
    #     pattern = r'\((.*?), (.*?), (\d+(?:\.\d+)?\s*#\s*\d+(?:\.\d+)?)\)'
    #     matches = re.findall(pattern, decoded_text)
    #     return [{"Aspect": a.strip(), "Opinion": o.strip(), "VA": v.replace(" ", "")} for a, o, v in matches]
    #
    # model.eval()
    # with open(PREDICT_FILE, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    #     for line in tqdm(fin, desc="Predicting"):
    #         item = json.loads(line)
    #         dynamic_examples = retriever.retrieve(item["Text"], k=3)
    #         prompt = build_infer_prompt(item["Text"], dynamic_examples)
    #         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    #
    #         with torch.no_grad():
    #             outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False,
    #                                      pad_token_id=tokenizer.eos_token_id)
    #
    #         # Extract only the newly generated part
    #         decoded = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    #         fout.write(json.dumps({
    #             "ID": item["ID"],
    #             "Text": item["Text"],
    #             "Triplet": extract_triplets_regex(decoded)
    #         }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()