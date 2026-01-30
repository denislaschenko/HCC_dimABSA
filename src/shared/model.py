import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from transformers import AutoModel, AutoConfig
from tqdm import tqdm

from torch import nn
from transformers import DebertaV2PreTrainedModel, DebertaV2Model
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from src.shared.utils import SupConLoss, CCCLoss, SafeLDLLoss

def nan_hook(module, input, output):
    """
    Checks every layer's output for NaNs.
    Stops training immediately if found.
    """
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f"\n!!!! NAN DETECTED IN LAYER: {module} !!!!")
            raise RuntimeError(f"NaN detected in {module}")
    elif isinstance(output, tuple):
        for i, x in enumerate(output):
            if isinstance(x, torch.Tensor) and torch.isnan(x).any():
                print(f"\n!!!! NAN DETECTED IN LAYER: {module} (Output {i}) !!!!")
                raise RuntimeError(f"NaN detected in {module}")

class TransformerVARegressor(nn.Module):
    def __init__(self, model_name: str, num_bins: int, dropout: float = 0.1):
        super().__init__()

        print(f"Loading model: {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config, use_safetensors=True)
        print("--- Checking Weights for NaNs ---")
        for name, param in self.backbone.named_parameters():
            if torch.isnan(param).any():
                print(f"CRITICAL ERROR: Found NaNs in pretrained weight: {name}")
                raise RuntimeError("The downloaded model checkpoint is corrupted. Clear cache and retry.")
        print("Weights are clean.")

        for name, layer in self.backbone.named_modules():
            layer.register_forward_hook(nan_hook)

        self.dropout = nn.Dropout(dropout)


        combined_size = self.config.hidden_size * 4
        self.v_head = nn.Linear(combined_size, num_bins)
        self.a_head = nn.Linear(combined_size, num_bins)

        # self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)

    def mean_pooling(self, last_hidden_state, attention_mask):
        """
        Aggregates token embeddings into a single sentence embedding using Mean Pooling.
        Essential for DeBERTa performance.
        """
        # attention_mask shape: [batch, seq_len] -> expand to [batch, seq_len, hidden]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())

        last_hidden_state = last_hidden_state.masked_fill(~mask_expanded.bool(), 0.0)

        # Sum embeddings where mask is 1
        sum_embeddings = torch.sum(last_hidden_state, 1)
        # Count tokens where mask is 1 (avoid div by zero with clamp)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        concatenate_pooling = torch.cat([outputs.hidden_states[i] for i in [-1, -2, -3, -4]], dim=-1)
        pooled_output = self.mean_pooling(concatenate_pooling, attention_mask)
        x = self.dropout(pooled_output)

        v_logits = self.v_head(x)
        a_logits = self.a_head(x)

        return v_logits, a_logits

    def train_epoch(self, dataloader, optimizer, scheduler, device, loss_fn, scaler):
        self.train()
        total_loss = 0
        valid_batches = 0

        ccc_loss_fn = CCCLoss().to(device)
        bins = torch.linspace(1.0, 9.0, 9).to(device)

        for batch in tqdm(dataloader, desc="Training Epoch"):
            input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            orig_scores = batch["orig_scores"].to(device)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                v_logits, a_logits = self(input_ids, attention_mask)
                loss_ldl = loss_fn(v_logits, orig_scores[:, 0]) + loss_fn(a_logits, orig_scores[:, 1])

                v_probs, a_probs = torch.softmax(v_logits, dim=1), torch.softmax(a_logits, dim=1)
                v_pred = torch.sum(v_probs * bins, dim=1)
                a_pred = torch.sum(a_probs * bins, dim=1)
                loss_ccc = ccc_loss_fn(v_pred, orig_scores[:, 0]) + ccc_loss_fn(a_pred, orig_scores[:, 1])

                loss = loss_ldl + (0.5 * loss_ccc)

            if torch.isnan(loss):
                continue

            # Scale and Backprop
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # FIX: Added missing loss increment
            total_loss += loss.item()
            valid_batches += 1

        return total_loss / valid_batches if valid_batches > 0 else 0.0

    def eval_epoch(self, dataloader, loss_fn, device):
        self.eval()
        total_loss = 0
        valid_batches = 0

        ccc_loss_fn = CCCLoss().to(device)
        bins = torch.linspace(1.0, 9.0, 9).to(device)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                orig_scores = batch["orig_scores"].to(device)


                with autocast(device_type="cuda"):
                    v_logits, a_logits = self(input_ids, attention_mask)

                    loss_ldl = loss_fn(v_logits, orig_scores[:, 0]) + loss_fn(a_logits, orig_scores[:, 1])

                    v_probs, a_probs = torch.softmax(v_logits, dim=1), torch.softmax(a_logits, dim=1)
                    v_pred = torch.sum(v_probs * bins, dim=1)
                    a_pred = torch.sum(a_probs * bins, dim=1)
                    loss_ccc = ccc_loss_fn(v_pred, orig_scores[:, 0]) + ccc_loss_fn(a_pred, orig_scores[:, 1])

                    loss = loss_ldl + (0.5 * loss_ccc)

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    valid_batches += 1

        return total_loss / valid_batches if valid_batches > 0 else float('nan')