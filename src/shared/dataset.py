import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.shared import config


def generate_label_distribution(target_value, min_val=1.0, max_val=9.0, num_bins=config.NUM_BINS, sigma=1.0):
    bin_centers = np.linspace(min_val, max_val, num_bins)
    probs = np.exp(-(bin_centers - target_value)**2 / (2 * sigma**2))
    probs = probs / np.sum(probs)

    return torch.tensor(probs, dtype=torch.float32)

class VADataset(Dataset):
    def __init__(self, dataframe, tokenizer: PreTrainedTokenizer, max_len: int, num_bins: int, sigma: float):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.num_bins = num_bins
        self.sigma = sigma

        if config.current_config.get("INCLUDE_OPINION"):
            self.opinions = dataframe["Opinion"].tolist()
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()

        if "Valence" in dataframe.columns and "Arousal" in dataframe.columns:
            self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)
        else:
            self.labels = np.zeros((len(dataframe), 2), dtype=float)

        aspects = dataframe["Aspect"].tolist()
        sentences = dataframe["Text"].tolist()

        include_opinion = config.current_config.get("INCLUDE_OPINION", False)
        has_opinion_col = "Opinion" in dataframe.columns

        if include_opinion and has_opinion_col:
            opinions = dataframe["Opinion"].tolist()
        else:
            opinions = None

        full_texts = []
        sep = self.tokenizer.sep_token

        for i in range(len(sentences)):
            if opinions:
                text = f"{aspects[i]} {sep} {opinions[i]} {sep} {sentences[i]}"
            else:
                text = f"{aspects[i]} {sep} {sentences[i]}"
            full_texts.append(text)

        print(f"Tokenizing {len(full_texts)} samples...")
        self.encodings = self.tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}

        val_score = self.labels[idx][0]
        aro_score = self.labels[idx][1]

        val_dist = generate_label_distribution(val_score, num_bins=self.num_bins, sigma=self.sigma)
        aro_dist = generate_label_distribution(aro_score, num_bins=self.num_bins, sigma=self.sigma)

        item["labels"] = torch.stack([val_dist, aro_dist])
        item["orig_scores"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item
