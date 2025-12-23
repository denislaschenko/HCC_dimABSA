import json
import argparse
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from src.subtask_1.train_subtask1 import main as train_reg
from src.shared import config

TRAIN_DATA_PATH = config.DATA_DIR
NUM_SHOTS = 3


BASE_INSTRUCTION = """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, O) tuples, where:
- A is an Aspect term (a phrase describing an entity mentioned in "Text")
- O is an Opinion term
- You must always format the output as a valid JSON object.
- If no relevant Aspect or Opinion exists, output an empty list for "Triplet".
"""

input_file = config.PREDICT_FILE
output_file = config.PREDICTION_FILE


class ExampleRetriever:
    """Handles loading training data and retrieving similar examples."""

    def __init__(self, train_path, model_name='all-MiniLM-L6-v2'):
        print(f"Loading retrieval model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.examples = []
        self.embeddings = None

        print(f"Loading reference examples from {train_path}...")
        try:
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))
        except FileNotFoundError:
            print(f"WARNING: Train file {train_path} not found. RAG will fail.")
            return

        texts = [ex["Text"] for ex in self.examples]
        print(f"Embedding {len(texts)} examples... (this happens once)")
        self.embeddings = self.encoder.encode(texts, convert_to_tensor=True)

    def retrieve(self, query_text, k=3):
        """Finds k most similar examples to the query text."""
        if self.embeddings is None:
            return []

        query_emb = self.encoder.encode(query_text, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_results = torch.topk(scores, k=k)

        retrieved_examples = []
        for idx in top_results.indices:
            retrieved_examples.append(self.examples[idx])

        return retrieved_examples


def build_dynamic_prompt(text: str, retrieved_examples: list):
    """Constructs the ICL prompt using retrieved examples."""
    prompt_str = BASE_INSTRUCTION + "\n### Examples:\n"

    for ex in retrieved_examples:

        triplets = ex.get("Triplet", [])

        sanitized_triplets = []
        for t in triplets:
            sanitized_triplets.append({
                "Aspect": t["Aspect"],
                "Opinion": t["Opinion"],
                "VA": "0#0"
            })

        input_json = json.dumps({"ID": ex["ID"], "Text": ex["Text"]}, ensure_ascii=False)
        output_json = json.dumps({"ID": ex["ID"], "Text": ex["Text"], "Triplet": sanitized_triplets},
                                 ensure_ascii=False)

        prompt_str += f"Input:\n{input_json}\n\nOutput:\n{output_json}\n\n"

    target_input = json.dumps({"ID": "test_id", "Text": text}, ensure_ascii=False)
    prompt_str += f"### Question:\nNow complete the following example, never change the Layout described in the Output examples and always predict \"VA\" to be \"0#0\":\nInput:\n{target_input}\n\nOutput:"

    return f"<|user|>\n{prompt_str}\n<|assistant|>\n"


def extract_pairs_from_llm_output(output_text: str):
    try:
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = output_text[start:end]
            data = json.loads(json_str)
            triplets = data.get("Triplet", [])
            pairs = []
            for t in triplets:
                if "Aspect" in t and "Opinion" in t:
                    pairs.append((t["Aspect"], t["Opinion"]))
            return pairs
    except json.JSONDecodeError:
        pass

    pattern = r'"Aspect":\s*"([^"]+)"\s*,\s*"Opinion":\s*"([^"]+)"'
    matches = re.findall(pattern, output_text)
    return matches


def process_jsonl(model_name, input_path, output_path, train_path):
    retriever = ExampleRetriever(train_path, model_name='all-MiniLM-L6-v2')

    print(f"Loading Generation Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    model.eval()

    print(f"Reading input: {input_path}")
    try:
        with open(input_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    print(f"Writing output: {output_path}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(lines, desc="Extracting"):
            if not line.strip(): continue

            item = json.loads(line)
            text = item["Text"]

            examples = retriever.retrieve(text, k=NUM_SHOTS)

            prompt = build_dynamic_prompt(text, examples)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs.input_ids.shape[1]
            generated_tokens = generated[0][input_len:]
            llm_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            pairs = extract_pairs_from_llm_output(llm_output)

            formatted_triplets = []
            for aspect, opinion in pairs:
                formatted_triplets.append({
                    "Aspect": aspect.strip(),
                    "Opinion": opinion.strip(),
                    "VA": "0#0"
                })

            new_entry = {
                "ID": item["ID"],
                "Text": text,
                "Triplet": formatted_triplets
            }

            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            fout.flush()

    print("Done.")

    train_reg(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
    args = parser.parse_args()

    process_jsonl(args.model, input_file, output_file, TRAIN_DATA_PATH)


if __name__ == "__main__":
    main()