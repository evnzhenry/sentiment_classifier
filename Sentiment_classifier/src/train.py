"""Training loop with early stopping, gradient clipping, and optional mixed precision."""
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, model, optimizer, scheduler=None, device=None, grad_clip=1.0, amp=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_clip = grad_clip
        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler() if (self.amp and self.device=='cuda') else None
        self.model.to(self.device)

    def train_epoch(self, dataloader, loss_fn):
        self.model.train()
        losses = []
        for batch in tqdm(dataloader, desc='train', leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            if self.amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss if hasattr(outputs, 'loss') else loss_fn(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else loss_fn(outputs, labels)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            losses.append(loss.item())
        return np.mean(losses)

    @torch.no_grad()
    def eval_epoch(self, dataloader):
        self.model.eval()
        preds = []
        trues = []
        losses = []
        for batch in tqdm(dataloader, desc='eval', leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(batch_preds.tolist())
            trues.extend(labels.cpu().numpy().tolist())
            if loss is not None:
                losses.append(loss.item())
        return preds, trues, np.mean(losses) if losses else None

    def fit(self, train_loader, val_loader, loss_fn, epochs=3, patience=2, save_path='checkpoint.pt'):
        best_f1 = -1.0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        for epoch in range(1, epochs+1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader, loss_fn)
            preds, trues, val_loss = self.eval_epoch(val_loader)
            val_f1 = f1_score(trues, preds, average='macro')
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} time={time.time()-t0:.1f}s")
            # early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # save model
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break
        return history


if __name__ == '__main__':
    print('Trainer utility ready')
