import torch
import torch.nn as nn
from transformers import AutoModel
from tqdm import tqdm

from torch import nn
from transformers import DebertaV2PreTrainedModel, DebertaV2Model
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from src.shared.utils import SupConLoss

class TransformerVARegressor(nn.Module):
    def __init__(self, model_name: str, num_bins: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        self.v_head = nn.Linear(self.backbone.config.hidden_size, num_bins)
        self.a_head = nn.Linear(self.backbone.config.hidden_size, num_bins)

        # self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_output)

        v_logits = self.v_head(x)
        a_logits = self.a_head(x)

        return v_logits, a_logits
        # return self.reg_head(x)

    def train_epoch(self, dataloader, optimizer, scheduler, device, loss_fn):
        self.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training Epoch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            #  outputs = self(input_ids, attention_mask)
            #  loss = loss_fn(outputs, labels)
            v_logits, a_logits = self(input_ids, attention_mask)

            loss_v = loss_fn(v_logits, labels[:, 0])
            loss_a = loss_fn(a_logits, labels[:, 1])

            loss = loss_v + loss_a

            loss.backward()
            optimizer.step()
            scheduler.step() # Update learning rate

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def eval_epoch(self, dataloader, loss_fn, device):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                v_logits, a_logits = self(input_ids, attention_mask)

                loss_v = loss_fn(v_logits, labels[:, 0])
                loss_a = loss_fn(a_logits, labels[:, 1])

                # outputs = self(input_ids, attention_mask)
                # loss = loss_fn(outputs, labels)
                # total_loss += loss.item()

                total_loss += (loss_v + loss_a).item()
        return total_loss / len(dataloader)
