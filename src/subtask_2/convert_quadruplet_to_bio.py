import re
import json
import pandas as pd
from transformers import AutoTokenizer

def bio_tags_for_span(tokens, span_tokens):
    tags = ["O"] * len(tokens)
    span_str = " ".join(span_tokens)

    for i in range(len(tokens)):
        window = " ".join(tokens[i:i+len(span_tokens)])
        if window == span_str:
            tags[i] = "B"
            for j in range(1, len(span_tokens)):
                tags[i+j] = "I"
            return tags
    return tags

def convert_quadruplet_to_bio(input_path, output_path, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    out = []
    with open(input_path) as f:
        for line in f:
            item = json.loads(line)
            text = item["Text"]

            encoded = tokenizer(text, return_offsets_mapping=True)
            tokens = tokenizer.tokenize(text)

            aspect_tags = ["O"] * len(tokens)
            opinion_tags = ["O"] * len(tokens)
            valence_seq = [None] * len(tokens)
            arousal_seq = [None] * len(tokens)

            for q in item["Quadruplet"]:
                asp = q["Aspect"]
                opn = q["Opinion"]

                if asp == "NULL" or opn == "NULL":
                    continue

                V, A = map(float, q["VA"].split("#"))
                asp_tokens = tokenizer.tokenize(asp)
                opn_tokens = tokenizer.tokenize(opn)

                asp_bio = bio_tags_for_span(tokens, asp_tokens)
                opn_bio = bio_tags_for_span(tokens, opn_tokens)

                for i, tag in enumerate(asp_bio):
                    if tag != "O":
                        aspect_tags[i] = tag
                        valence_seq[i] = V
                        arousal_seq[i] = A

                for i, tag in enumerate(opn_bio):
                    if tag != "O":
                        opinion_tags[i] = tag
                        valence_seq[i] = V
                        arousal_seq[i] = A

            out.append({
                "ID": item["ID"],
                "Text": text,
                "AspectBIO": aspect_tags,
                "OpinionBIO": opinion_tags,
                "ValenceSeq": valence_seq,
                "ArousalSeq": arousal_seq
            })

    with open(output_path, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    print("Saved:", output_path)
