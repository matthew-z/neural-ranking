import csv
import gzip
import os
import subprocess

import torch
import transformers
from torch.utils import data
from tqdm import tqdm


class MsMarcoPassageTriplesDataset(data.Dataset):
    """Dataset from passage triples tsv."""
    def __init__(self, ms_marco_path):
        self.ms_marco_path = ms_marco_path
        self.triplepath = os.path.join(ms_marco_path, "triples.train.small.tsv")
        self._len =  get_line_nums_of_file(self.triplepath)

    def __getitem__(self, item):
        query, pos_passage, neg_passage = get_triplet(self.triplepath, item)
        pos_passage = truncate_untokenized_doc(pos_passage)
        neg_passage = truncate_untokenized_doc(neg_passage)
        return query, pos_passage, neg_passage

    def __len__(self):
        return self._len

    class Collater():
        def __init__(self, bert_model: str = None):
            self.encoder = transformers.BertTokenizer.from_pretrained(bert_model or "bert-base-uncased")

        def __call__(self, examples):
            inputs = []
            labels = []

            for query, pos_passage, neg_passage in examples:
                inputs.append((query, pos_passage))
                labels.append(1)

                inputs.append((query, neg_passage))
                labels.append(0)

            x = self.encoder.batch_encode_plus(
                    inputs,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=512,
                    pad_to_max_length=True,
                    truncation_strategy="only_second"
            )
            batch = x, torch.LongTensor(labels)
            return batch


class MsMarcoPassageTriplesIterableDataset(data.IterableDataset):
    """Iterable dataset from passage triples tsv."""
    def __init__(self, ms_marco_path):
        self.ms_marco_path = ms_marco_path
        self.triplepath = os.path.join(ms_marco_path, "triples.train.small.tsv")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, skip = 0, 0
        else:
            start, skip = worker_info.id, worker_info.num_workers - 1
        with open(self.triplepath, 'r') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for _ in range(start):
                next(tsvreader)
            yield next(tsvreader)
            while True:
                for _ in range(skip):
                    next(tsvreader)
                yield next(tsvreader)

    class Collater():
        def __init__(self, bert_model: str = None):
            self.encoder = transformers.BertTokenizer.from_pretrained(bert_model or "bert-base-uncased")

        def __call__(self, examples):
            inputs = []
            labels = []

            for query, pos_passage, neg_passage in examples:
                inputs.append((query, pos_passage))
                labels.append(1)

                inputs.append((query, neg_passage))
                labels.append(0)

            x = self.encoder.batch_encode_plus(
                    inputs,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=512,
                    pad_to_max_length=True,
                    truncation_strategy="only_second"
            )
            batch = x, torch.LongTensor(labels)
            return batch


class MsMarcoPassageReRankDataset(data.Dataset):
    def __init__(self, ms_marco_path):
        self.ms_marco_path = ms_marco_path
        self.top1000_path = os.path.join(ms_marco_path, "msmarco-passagetest2019-top1000.tsv.gz")
        self.qrel_path = os.path.join(ms_marco_path, "2019qrels-pass.txt")
        self._index = self.get_test_index()

    def get_test_index(self):
        rels = get_qrels(self.qrel_path)
        index = []
        with gzip.open(self.top1000_path, 'rt', encoding='utf-8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for qid, pid, query, passage in tsvreader:
                if qid not in rels:  # some queries not in qrels
                    continue
                label = rels[qid].get(pid, 0)
                index.append((qid, pid, query, passage, label))
        return index

    def __getitem__(self, item):
        return self._index[item]

    def __len__(self):
        return len(self._index)

    class Collater():
        def __init__(self, bert_model: str = None):
            self.encoder = transformers.BertTokenizer.from_pretrained(bert_model or "bert-base-uncased")

        def __call__(self, examples):
            fields = list(zip(*examples))
            if len(fields) == 5:
                qid, pid, query, passage, labels = fields
            else:
                raise ValueError("incorrect length of fields, expected 3, found %d" % len(fields))

            x = self.encoder.batch_encode_plus(
                zip(query, passage),
                return_tensors="pt",
                add_special_tokens=True,
                max_length=512,
                pad_to_max_length=True,
                truncation_strategy="only_second"
            )
            return qid, pid, x, torch.LongTensor(labels)


def get_triplet(path, idx):
    with open(path, 'r') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(tsvreader):
            if i == idx:
                return row
    raise ValueError('Cannot get index {} in file.'.format(idx))


def get_qrels(path):
    """Read TREC-style qrel text file, returns a nested dict mapping qid to pid to rel score."""
    rels = {}
    with open(path) as f:
        for line in f:
            qid, _, pid, rel = line.split()
            rel = int(rel)
            if rel > 0:
                rels.setdefault(qid, {}) 
                rels[qid][pid] = rel
    return rels


def get_line_nums_of_file(filepath):
    result = subprocess.getoutput("wc -l %s" % filepath)
    if result:
        length = int(result.split()[0])
    else:
        raise ValueError("Cannot fetch the length of %s" % filepath)
    return length


def truncate_untokenized_doc(body):
    return body[:5000]
