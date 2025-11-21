import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from src.subtask_2.dataset import ASTEDataset
from src.subtask_2.model import DimASTEModel
from src.subtask_2 import config
from src.shared import utils

def compute_loss(asp_logits, opn_logits, va_pred, batch):
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    mse = nn.MSELoss()

    loss_asp = ce(asp_logits.view(-1, 3), batch["labels_aspect"].view(-1))
    loss_opn = ce(opn_logits.view(-1, 3), batch["labels_opinion"].view(-1))

    mask = batch["attention_mask"].unsqueeze(-1)
    loss_va = mse(va_pred * mask, torch.stack([batch["valence"], batch["arousal"]], dim=-1) * mask)

    return loss_asp + loss_opn + 2 * loss_va

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        asp_logits, opn_logits, va_pred = model(batch["input_ids"], batch["attention_mask"])
        loss = compute_loss(asp_logits, opn_logits, va_pred, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total += loss.item()

    return total / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            asp_logits, opn_logits, va_pred = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_loss(asp_logits, opn_logits, va_pred, batch)
            total += loss.item()
    return total / len(loader)

def main():
    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_df = utils.jsonl_to_df(utils.load_jsonl(config.PROCESSED_TRAIN))
    dev_df   = utils.jsonl_to_df(utils.load_jsonl(config.PROCESSED_DEV))

    train_set = ASTEDataset(train_df, tokenizer, config.MAX_LEN)
    dev_set   = ASTEDataset(dev_df, tokenizer, config.MAX_LEN)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_set, batch_size=config.BATCH_SIZE)

    model = DimASTEModel(config.MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=config.LR)

    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1*total_steps), total_steps
    )

    best = 1e9

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        tr = train_epoch(model, train_loader, optimizer, scheduler, device)
        dv = eval_epoch(model, dev_loader, device)
        print(f"Train {tr:.4f} | Dev {dv:.4f}")

        if dv < best:
            best = dv
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("Saved best model.")

if __name__ == "__main__":
    main()
