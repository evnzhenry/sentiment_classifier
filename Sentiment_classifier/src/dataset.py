"""PyTorch Dataset and collate function for tokenized inputs."""
import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.tokenizer is not None:
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
            item = {k: v.squeeze(0) for k, v in enc.items()}
        else:
            item = {'text': text}
        if self.labels is not None:
            item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def collate_fn(batch):
    # batch is a list of dicts
    if 'input_ids' in batch[0]:
        input_ids = torch.stack([b['input_ids'] for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        item = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'token_type_ids' in batch[0]:
            token_type_ids = torch.stack([b['token_type_ids'] for b in batch])
            item['token_type_ids'] = token_type_ids
        if 'labels' in batch[0]:
            labels = torch.stack([b['labels'] for b in batch])
            item['labels'] = labels
        return item
    else:
        # return texts and labels
        texts = [b['text'] for b in batch]
        labels = torch.tensor([b['labels'] for b in batch]) if 'labels' in batch[0] else None
        return {'texts': texts, 'labels': labels}


if __name__ == '__main__':
    print('Dataset utilities ready')
