import os
import torch
from torchtext import data
import transformers
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


textField = lambda: data.Field()
fields = [
    ('docid', data.RawField()),
    ('query', textField()),

    ('posdoc_id', data.RawField()),
    ('posdoc_url', data.RawField()),
    ('posdoc_title', textField()),
    ('posdoc_body', textField()),

    ('negdoc_id', data.RawField()),
    ('negdoc_url', data.RawField()),
    ('negdoc_title', textField()),
    ('negdoc_body', textField()),

]

pos = data.TabularDataset(path='/Users/zhaohaozeng/ir/collection/triples.tsv.small', format='tsv',fields=fields)