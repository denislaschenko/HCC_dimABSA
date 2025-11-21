import json
import torch
from transformers import AutoTokenizer

from subtask_2.model import DimASTEModel
from subtask_2.extractor import extract_triplets
from subtask_2 import config

def run_inference(input_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    model = DimASTEModel(config.MODEL_NAME)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()

    data = [json.loads(l) for l in open(input_path)]

    out = []
    for row in data:
        text = row["Text"]

        enc = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            asp_logits, opn_logits, va_pred = model(enc["input_ids"], enc["attention_mask"])

        aspect_labels = asp_logits.argmax(-1).squeeze().tolist()
        opinion_labels = opn_logits.argmax(-1).squeeze().tolist()

        triplets = extract_triplets(
            text,
            tokenizer,
            aspect_labels,
            opinion_labels,
            va_pred.squeeze()
        )

        out.append({"ID": row["ID"], "Triplet": triplets})

    with open(output_path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print("Saved:", output_path)
