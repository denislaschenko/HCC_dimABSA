# extract triplets
import re
from typing import List, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PAIR_REGEX = re.compile(r'\("(.+?)"\s*,\s*"(.+?)"\)')

def build_icl_prompt(examples: List[Dict], query_text: str, example_format: str = "pairlist") -> str:
    """
    Build a simple prompt with examples then the query.
    - examples: list of dicts with 'Text' and 'Quadruplet' or 'Triplet' fields.
      For each example we produce Triplet: [("asp","opn"), ...]
    - query_text: new text to extract from
    """
    parts = []
    for ex in examples:
        text = ex.get("Text", "").strip()
        triplets = ex.get("Quadruplet") or ex.get("Triplet") or []
        pairs = []
        for q in triplets:
            asp = q.get("Aspect", "").strip()
            opn = q.get("Opinion", "").strip()
            if not asp or asp.upper() == "NULL" or not opn or opn.upper() == "NULL":
                continue
            pairs.append(f'("{asp}", "{opn}")')
        parts.append(f"Text: {text}\nTriplet: [{', '.join(pairs)}]\n")
    parts.append(f"Text: {query_text}\nTriplet: [")
    return "\n".join(parts)

def parse_triplets_from_generated(text: str) -> List[Tuple[str,str]]:
    """
    Parse patterns like ("aspect", "opinion") from the generator output.
    Returns list of (aspect, opinion)
    """
    pairs = PAIR_REGEX.findall(text)
    return [(a.strip(), o.strip()) for a,o in pairs]

class ICLGenerator:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate_pairs(self,
                       examples: List[Dict],
                       query_text: str,
                       max_length: int = 128,
                       temperature: float = 0.7,
                       top_k: int = 50,
                       num_return_sequences: int = 1) -> List[Tuple[str,str]]:
        prompt = build_icl_prompt(examples, query_text)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        gen = self.model.generate(
            **enc,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            num_return_sequences=num_return_sequences
        )
        out = []
        # take first returned sequence
        gen_text = self.tokenizer.decode(gen[0], skip_special_tokens=True)
        pairs = parse_triplets_from_generated(gen_text)
        return pairs
