import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TransformerVARegressor(nn.Module):
    def __init__(self, model_name: str, num_bins: int, dropout: float = 0.3):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        self.dropout = nn.Dropout(dropout)

        combined_size = self.config.hidden_size * 4

        self.v_head = nn.Linear(combined_size, num_bins)
        self.a_head = nn.Linear(combined_size, num_bins)

    def _mean_pooling(self, last_hidden_state, attention_mask):
        """
        Aggregates token embeddings into a single sentence embedding.
        Masks padding tokens to ensure they don't affect the average.
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())

        last_hidden_state = last_hidden_state.masked_fill(~mask_expanded.bool(), 0.0)

        sum_embeddings = torch.sum(last_hidden_state, 1)

        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        all_states = outputs.hidden_states
        concatenate_pooling = torch.cat([all_states[i] for i in [-1, -2, -3, -4]], dim=-1)

        pooled_output = self._mean_pooling(concatenate_pooling, attention_mask)
        x = self.dropout(pooled_output)

        v_logits = self.v_head(x)
        a_logits = self.a_head(x)

        return v_logits, a_logits