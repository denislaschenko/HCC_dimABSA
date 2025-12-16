import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.subtask_1 import config


def generate_label_distribution(target_value, min_val=1.0, max_val=9.0, num_bins=config.NUM_BINS, sigma=1.0):
    bin_centers = np.linspace(min_val, max_val, num_bins)
    probs = np.exp(-(bin_centers - target_value)**2 / (2 * sigma**2))
    probs = probs / np.sum(probs)

    return torch.tensor(probs, dtype=torch.float32)

class VADataset(Dataset):

    def __init__(self, dataframe, tokenizer: PreTrainedTokenizer, max_len: int, num_bins: int, sigma: float):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()

        if "Valence" in dataframe.columns and "Arousal" in dataframe.columns:
            self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)
        else:
            import numpy as np
            self.labels = np.zeros((len(dataframe), 2), dtype=float)
        
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.num_bins = num_bins
        self.sigma = sigma

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Format: "Aspect: [SEP] Text"
        sep_token = self.tokenizer.sep_token
        text = f"{self.aspects[idx]} {sep_token} {self.sentences[idx]}"

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        val_score = self.labels[idx][0]
        aro_score = self.labels[idx][1]

        val_dist = generate_label_distribution(val_score, num_bins=self.num_bins, sigma=self.sigma)
        aro_dist = generate_label_distribution(aro_score, num_bins=self.num_bins, sigma=self.sigma)

        label_dist = torch.stack([val_dist, aro_dist])

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": label_dist,
            "orig_scores": torch.tensor(self.labels[idx], dtype=torch.float)
        }
