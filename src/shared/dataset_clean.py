import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def generate_label_distribution(target_value, min_val=1.0, max_val=9.0, num_bins=9, sigma=1.0):
    """
    Generates a Gaussian distribution over the bins centered at target_value.
    Used for Label Distribution Learning (LDL).
    """
    bin_centers = np.linspace(min_val, max_val, num_bins)
    probs = np.exp(-(bin_centers - target_value) ** 2 / (2 * sigma ** 2))
    probs = probs / np.sum(probs)  # Normalize to sum to 1

    return torch.tensor(probs, dtype=torch.float32)


class VADataset(Dataset):
    def __init__(self,
                 dataframe,
                 tokenizer: PreTrainedTokenizer,
                 max_len: int,
                 num_bins: int,
                 sigma: float,
                 include_opinion: bool = False):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_bins = num_bins
        self.sigma = sigma
        self.include_opinion = include_opinion

        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()

        if self.include_opinion and "Opinion" in dataframe.columns:
            self.opinions = dataframe["Opinion"].tolist()
        else:
            self.opinions = None

        if "Valence" in dataframe.columns and "Arousal" in dataframe.columns:
            self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)
        else:
            self.labels = np.zeros((len(dataframe), 2), dtype=float)

        self.encodings = self._tokenize_data()

    def _tokenize_data(self):
        text_a = []
        text_b = []

        print(f"Tokenizing {len(self.sentences)} pairs...")

        for i in range(len(self.sentences)):
            if self.opinions is not None:
                text_a.append(f"{self.aspects[i]}, {self.opinions[i]}")
            else:
              text_a.append(str(self.aspects[i]))

            text_b.append(str(self.sentences[i]))

        return self.tokenizer(
            text_a,
            text_b,
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