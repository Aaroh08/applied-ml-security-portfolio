"""Shared text normalization and frozen, CPU-only DistilBERT embeddings."""
import re
import unicodedata
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
LABELS=['atypical','bipolar','major depressive','no','postpartum','psychotic']
DISCLAIMER='Research classification of dataset tags, not a diagnosis or assessment of a person. All scores are uncertain and uncalibrated.'
LABEL_PATTERN=re.compile(r'\b(?:atypical|bipolar|major\s+depress\w*|depress\w*|postpartum|post\s+partum|psychot\w*)\b',re.I)

def clean_text(text):
    text=unicodedata.normalize('NFKC',text).lower()
    text=re.sub(r'https?://\S+|www\.\S+',' url ',text)
    text=re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b',' email ',text)
    text=re.sub(r'(?<!\w)@[\w]+',' user ',text)
    return ' '.join(text.split())

def duplicate_key(text):
    return re.sub(r'[^a-z0-9]+','',clean_text(text))

def mask_label_terms(text):
    return LABEL_PATTERN.sub(' topic ',text)

def validate(payload):
    if not isinstance(payload,dict) or set(payload)!={'text'}:
        raise ValueError('Provide exactly one JSON field: text.')
    text=payload['text']
    if not isinstance(text,str) or len(text)>4000 or len(text.strip())<10:
        raise ValueError('text must contain 10 to 4000 characters.')
    if len(re.findall(r'[a-z]+',clean_text(text)))<3:
        raise ValueError('Provide at least three English words for this English-only research model.')
    return clean_text(text)

class Embedder:
    def __init__(self,path=None):
        import torch
        from transformers import AutoTokenizer, AutoModel
        torch.set_num_threads(4)
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(str(path or ROOT/'.cache/distilbert'),local_files_only=True)
        self.model=AutoModel.from_pretrained(str(path or ROOT/'.cache/distilbert'),local_files_only=True).eval()
        self.model.requires_grad_(False)
        self.max_length=128

    def encode(self,texts,batch_size=8,progress=False):
        outputs=[]; truncated=0
        for start in range(0,len(texts),batch_size):
            batch=list(texts[start:start+batch_size])
            lengths=self.tokenizer(batch,truncation=False,add_special_tokens=True)['input_ids']
            truncated+=sum(len(x)>self.max_length for x in lengths)
            tokens=self.tokenizer(batch,padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
            with self.torch.inference_mode():
                hidden=self.model(**tokens).last_hidden_state
                mask=tokens['attention_mask'].unsqueeze(-1)
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            outputs.append(pooled.cpu().numpy())
            if progress and start%400==0: print(f'Embedded {start+len(batch)}/{len(texts)}',flush=True)
        return np.concatenate(outputs),truncated
