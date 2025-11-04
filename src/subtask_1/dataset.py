import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class VADataset(Dataset):

    def __init__(self, dataframe, tokenizer: PreTrainedTokenizer, max_len: int):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.labels = dataframe[["Valence", "Arousal"]].values.astype(float)
        self.tokenizer = tokenizer
        self.max_len = max_len

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

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
