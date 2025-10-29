"""Model definitions: BiLSTM baseline and Transformer wrapper for fine-tuning."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=None, embedding_dim=300, hidden_dim=256, num_layers=1, dropout=0.3, num_classes=3, embeddings=None):
        super().__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            if vocab_size is None:
                raise ValueError('vocab_size required if embeddings not provided')
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)
        emb = self.embedding(input_ids)
        outputs, (hn, cn) = self.bilstm(emb)
        # mean pooling
        pooled = outputs.mean(dim=1)
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits


class TransformerClassifier(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_labels=3):
        super().__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        # use AutoModelForSequenceClassification for convenience
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)


if __name__ == '__main__':
    print('Models module loaded')
