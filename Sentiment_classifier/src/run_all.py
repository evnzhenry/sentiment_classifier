"""Example orchestration script showing how to run EDA, preprocessing, tokenization, training and evaluation.

This is a convenience script — for full experiments, run from a notebook and adjust paths and hyperparameters.
"""
import os
from data_exploration import load_afrisenti, simple_eda
from preprocess import simple_clean
from tokenize_utils import get_tokenizer
from dataset import SentimentDataset, collate_fn
from models import TransformerClassifier
from train import Trainer
from eval_utils import compute_metrics, plot_confusion, compute_roc_auc
from torch.utils.data import DataLoader
import torch


def small_run(data_dir='data', model_key='xlm-roberta', max_length=128, batch_size=16, epochs=3):
    print('Loading data...')
    df = load_afrisenti(data_dir=data_dir)
    print('Running EDA...')
    stats = simple_eda(df, save_prefix=os.path.join(data_dir, 'eda'))

    # preprocess sample
    df['clean_text'] = df['text'].astype(str).apply(simple_clean)
    # map labels to 0/1/2 if not already
    label_map = {k: i for i, k in enumerate(sorted(df['label'].unique()))}
    df['label_id'] = df['label'].map(label_map)

    # quick stratified split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['label_id'], random_state=42)

    tokenizer = get_tokenizer(model_key)
    train_ds = SentimentDataset(train_df['clean_text'].tolist(), train_df['label_id'].tolist(), tokenizer=tokenizer, max_length=max_length)
    val_ds = SentimentDataset(val_df['clean_text'].tolist(), val_df['label_id'].tolist(), tokenizer=tokenizer, max_length=max_length)
    test_ds = SentimentDataset(test_df['clean_text'].tolist(), test_df['label_id'].tolist(), tokenizer=tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerClassifier(model_name=tokenizer.name_or_path, num_labels=len(label_map))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    trainer = Trainer(model, optimizer, device=None, grad_clip=1.0, amp=False)
    history = trainer.fit(train_loader, val_loader, loss_fn=None, epochs=epochs, patience=2, save_path='best.pt')

    # load best and evaluate
    model.load_state_dict(torch.load('best.pt'))
    preds, trues, _ = trainer.eval_epoch(test_loader)
    metrics = compute_metrics(trues, preds, labels=list(label_map.keys()))
    print('Test metrics:', metrics)
    plot_confusion(trues, preds, labels=list(label_map.keys()))


if __name__ == '__main__':
    # Run a small example. Note: ensure data is present in data/ before running.
    small_run()
