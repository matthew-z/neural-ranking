import csv
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
    def __init__(self, ms_marco_path, mode='dev'):
        self.ms_marco_path = ms_marco_path
        self.mode = mode
        if mode == 'dev':
            self.top1000_path = os.path.join(ms_marco_path, "top1000.dev")
            self.qrel_path = os.path.join(ms_marco_path, "qrels.dev.tsv")
        else:
            self.top1000_path = os.path.join(ms_marco_path, "msmarco-passagetest2019-top1000.tsv")
            qrel_path = os.path.join(ms_marco_path, "2019qrels-pass.txt")
            self.qrel_path = qrel_path if os.path.exists(qrel_path) else None
        self._index = self.get_index()

    def get_index(self):
        rels = get_qrels(self.qrel_path) if self.qrel_path else None
        d = get_top1000(self.top1000_path)
        queries = d['queries']
        passages = d['passages']
        top1000 = d['top1000']

        index = []
        for qid, pids in top1000.items():
            if (rels and qid not in rels) or len(pids) != 1000:
                continue
            query = queries[qid]
            for pid in pids:
                passage = passages[pid]
                if rels:
                    label = 1 if pid in rels[qid] else 0
                    index.append((qid, pid, query, passage, label))
                else:
                    index.append((qid, pid, query, passage))
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
            elif len(fields) == 4:
                qid, pid, query, passage = fields
                labels = None
            else:
                raise ValueError("incorrect length of fields, expected 4 or 5, found %d" % len(fields))

            x = self.encoder.batch_encode_plus(
                zip(query, passage),
                return_tensors="pt",
                add_special_tokens=True,
                max_length=512,
                pad_to_max_length=True,
                truncation_strategy="only_second"
            )
            if labels:
                return qid, pid, x, torch.LongTensor(labels)
            return qid, pid, x


def get_triplet(path, idx):
    with open(path, 'r') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(tsvreader):
            if i == idx:
                return row
    raise ValueError('Cannot get index {} in file.'.format(idx))


def get_qrels(path):
    """Read TREC-style qrel file, returns a nested dict mapping qid to pid to rel score."""
    ext = os.path.splitext(path)[1]
    res = {}
    with open(path) as f:
        if ext == '.txt':
            reader = map(lambda x: x.split(), f)
        elif ext == '.tsv':
            reader = csv.reader(f, delimiter='\t') 
        else:
            raise ValueError('Only txt or tsv allowed.')

        rels = _get_qrels(reader)
    return rels


def _get_qrels(it):
    rels = {}
    for qid, _, pid, rel in it:
        rel = int(rel)
        if rel > 0:
            rels.setdefault(qid, set()) 
            rels[qid].add(pid)
    return rels


def get_top1000(path):
    top1000 = {}
    passages = {}
    queries = {}
    with open(path) as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for qid, pid, query, passage in tsvreader:
            top1000.setdefault(qid, [])
            top1000[qid].append(pid)
            queries[qid] = query
            passages[pid] = passage
    return {
            'top1000': top1000,
            'passages': passages,
            'queries': queries
        }


def get_line_nums_of_file(filepath):
    result = subprocess.getoutput("wc -l %s" % filepath)
    if result:
        length = int(result.split()[0])
    else:
        raise ValueError("Cannot fetch the length of %s" % filepath)
    return length


def truncate_untokenized_doc(body):
    return body[:5000]
