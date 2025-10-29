"""Tokenizers for multilingual models (mBERT, XLM-R, AfriBERTa)

Provides helper to get tokenizer and to batch-tokenize texts with caching option.
"""
from transformers import AutoTokenizer

MODEL_MAP = {
    'mbert': 'bert-base-multilingual-cased',
    'xlm-roberta': 'xlm-roberta-base',
    # placeholder for AfriBERTa; user can replace with a checkpoint id
    'afriberta': 'microsoft/afriberta-base'
}

def get_tokenizer(model_key='xlm-roberta', use_fast=True):
    model_id = MODEL_MAP.get(model_key, model_key)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast)
    return tokenizer

def batch_tokenize(tokenizer, texts, max_length=128, truncation=True, padding='longest'):
    return tokenizer(texts, truncation=truncation, padding=padding, max_length=max_length)


if __name__ == '__main__':
    tok = get_tokenizer('xlm-roberta')
    print(tok(['This is a test', 'Ni jambo la habari']), max_length=10)
