import torch
import torch.nn as nn
from transformers import AutoModel

class DimASTEModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.aspect_head = nn.Linear(hidden, 3)
        self.opinion_head = nn.Linear(hidden, 3)
        self.va_head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask)
        seq = out.last_hidden_state

        return (
            self.aspect_head(seq),
            self.opinion_head(seq),
            self.va_head(seq),
        )
