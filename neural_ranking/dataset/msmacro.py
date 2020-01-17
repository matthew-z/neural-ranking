import csv
import gzip
import os
import subprocess

import torch
from torch import utils

import transformers
def combine_title_body(title, body, mode, ):
    if mode == "both":
        return title + " " + body
    elif mode == "title":
        return title
    else:
        return body


def get_line_nums_of_file(filepath):
    result = subprocess.getoutput("wc -l %s" % filepath)
    if result:
        length = int(result.split()[0])
    else:
        raise ValueError("Cannot fetch the length of %s" % filepath)
    return length


class MsMacroTriplesDataset(utils.data.IterableDataset):
    def __init__(self, triples_path, title_body_mode="both"):
        self.path = triples_path
        self.title_body_mode = title_body_mode
        self._len = 0

    def _combine_title_body(self, title, body):
        if self.title_body_mode == "both":
            return title + " " + body
        elif self.title_body_mode == "title":
            return title
        else:
            return body

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        with open(self.path, "rt") as f:
            for i, line in enumerate(f):
                if worker_info is not None and i % worker_info.num_workers != worker_info.id:
                    continue
                fields = line.split("\t")
                if len(fields) != 10:
                    continue
                [topicid, query, posdoc_id, posdoc_url, posdoc_title, posdoc_body,
                 negdoc_id, negdoc_url, negdoc_title, negdoc_body] = fields

                posdoc = combine_title_body(posdoc_title, posdoc_body, self.title_body_mode)
                negdoc = combine_title_body(negdoc_title, negdoc_body, self.title_body_mode)
                yield topicid, query, posdoc_id, posdoc, negdoc_id, negdoc

    def __len__(self):
        if self._len:
            return self._len
        self._len = get_line_nums_of_file(self.path)
        return self._len


    class Collater():
        def __init__(self, bert_model: str = None):
            self.encoder = transformers.BertTokenizer.from_pretrained(bert_model or "bert-base-uncased")

        def __call__(self, examples):
            inputs = []
            labels = []
            topic_ids = []
            queries = []

            doc_ids = []
            for topic_id, query, posdoc_id, posdoc, negdoc_id, negdoc in examples:
                doc_ids.append(posdoc_id)
                queries.append(query)
                inputs.append((query, posdoc))
                labels.append(1)

                doc_ids.append(negdoc_id)
                queries.append(query)
                inputs.append((query, negdoc))
                labels.append(0)

            x = self.encoder.batch_encode_plus(inputs,
                                               return_tensors="pt",
                                               add_special_tokens=True,
                                               max_length=512,
                                               truncation_strategy="only_second")
            return topic_ids, doc_ids, x, torch.LongTensor(labels)


class MsMacroReRankDataset(utils.data.IterableDataset):
    class Collater():
        def __init__(self, bert_model: str = None):
            self.encoder = transformers.BertTokenizer.from_pretrained(bert_model or "bert-base-uncased")

        def __call__(self, examples):
            fields = list(zip(*examples))
            if len(fields) == 5:
                topic_ids, doc_ids, queries, docs, labels = fields
            elif len(fields) == 4:
                topic_ids, doc_ids, queries, docs = fields
                labels = None
            else:
                raise ValueError("incorrect length of fields, "
                                 "expected 4 or 5, found %d" % len(fields))

            x = self.encoder.batch_encode_plus(zip(queries, docs),
                                               return_tensors="pt",
                                               add_special_tokens=True,
                                               max_length=512,
                                               truncation_strategy="only_second")
            if labels:
                return topic_ids, doc_ids, x, torch.LongTensor(labels)
            return topic_ids, doc_ids, x

    def __init__(self, msmacro_path, mode="dev", title_body_mode="both"):
        offset_path = os.path.join(msmacro_path, "msmarco-docs-lookup.tsv.gz")

        if mode == "dev":
            query_path = os.path.join(msmacro_path, "msmarco-docdev-queries.tsv.gz")
            top100_path = os.path.join(msmacro_path, "msmarco-docdev-top100.gz")
            qrel_path = os.path.join(msmacro_path, "msmarco-docdev-qrels.tsv.gz")
        else:
            query_path = os.path.join(msmacro_path, "msmarco-doctest-queries.tsv.gz")
            top100_path = os.path.join(msmacro_path, "msmarco-doctest-top100.gz")
            qrel_path = None

        self.queries = get_queries(query_path)
        self.qrels = get_qrels(qrel_path) if qrel_path else None
        self.docoffset = get_docoffset(offset_path)
        self.top100 = get_top100(top100_path)
        self.docs_path = os.path.join(msmacro_path, "msmarco-docs.tsv")
        self.title_body_mode = title_body_mode
        self._len = None

    def __len__(self):
        if self._len:
            return self._len
        n = 0
        for topic, docs in self.top100.items():
            n += len(docs)

        self._len = n
        return self._len

    def __iter__(self):
        i = 0
        worker_info = torch.utils.data.get_worker_info()
        with open(self.docs_path, "rt") as f:
            for topic_id, doc_ids in self.top100.items():
                query = self.queries[topic_id]
                for doc_id in doc_ids:
                    if worker_info is not None and i % worker_info.num_workers != worker_info.id:
                        continue
                    i += 1
                    fields = self._getcontent(doc_id, f)
                    if not fields:
                        continue
                    _, _, title, body = fields

                    doc = combine_title_body(title, body, self.title_body_mode)

                    if self.qrels:
                        label = doc_id in self.qrels[topic_id]
                        yield topic_id, doc_id, query, doc, label
                    else:
                        yield topic_id, doc_id, query, doc

    def _getcontent(self, docid, f):
        """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
        The content has four tab-separated strings: docid, url, title, body.
        """

        f.seek(self.docoffset[docid])
        line = f.readline()
        assert line.startswith(docid + "\t"), \
            f"Looking for {docid}, found {line}"
        fields = line.rstrip().split("\t")
        if len(fields) != 4:
            return None
        return fields


def get_top100(path):
    # The query string for each topicid is querystring[topicid]
    top100 = {}
    with gzip.open(path, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topic_id, _, docid, _, _, _] in tsvreader:
            top100.setdefault(topic_id, [])
            top100[topic_id].append(docid)

    return top100


def get_docoffset(offset_path):
    # In the corpus tsv, each docid occurs at offset docoffset[docid]
    docoffset = {}
    with gzip.open(offset_path, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)

    return docoffset


def get_queries(path):
    # The query string for each topicid is querystring[topicid]
    querystring = {}
    with gzip.open(path, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            querystring[topicid] = querystring_of_topicid

    return querystring


def get_qrels(path):
    rels = {}
    with gzip.open(path, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, _, docid, rel] in tsvreader:
            rels.setdefault(topicid, set())
            rel = int(rel)
            if rel > 0:
                rels[topicid].add(docid)
    return rels
