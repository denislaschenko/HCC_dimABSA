import torch
from torch.utils.data import Dataset

class ASTEDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["Text"]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        item = {k: v.squeeze() for k, v in enc.items()}

        asp = row["AspectBIO"][:self.max_len] + ["O"] * (self.max_len - len(row["AspectBIO"]))
        opn = row["OpinionBIO"][:self.max_len] + ["O"] * (self.max_len - len(row["OpinionBIO"]))

        tag2id = {"O":0, "B":1, "I":2}

        item["labels_aspect"]  = torch.tensor([tag2id[t] for t in asp], dtype=torch.long)
        item["labels_opinion"] = torch.tensor([tag2id[t] for t in opn], dtype=torch.long)

        valence = row["ValenceSeq"][:self.max_len] + [0]* (self.max_len - len(row["ValenceSeq"]))
        arousal = row["ArousalSeq"][:self.max_len] + [0]* (self.max_len - len(row["ArousalSeq"]))

        item["valence"] = torch.tensor(valence, dtype=torch.float)
        item["arousal"] = torch.tensor(arousal, dtype=torch.float)

        return item
