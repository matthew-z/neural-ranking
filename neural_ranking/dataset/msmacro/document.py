import csv
import gzip
import os
import subprocess

import torch
import transformers
from torch.utils import data
from tqdm import tqdm


class MsMacroDocTriplesDataset(data.Dataset):
    def __init__(self, ms_macro_path, title_body_mode="both"):
        self.ms_macro_path = ms_macro_path
        self.title_body_mode = title_body_mode
        self._index = []

        offset_path = os.path.join(ms_macro_path, "msmarco-docs-lookup.tsv.gz")
        query_path = os.path.join(ms_macro_path, "msmarco-doctrain-queries.tsv.gz")
        self.queries = get_queries(query_path)
        self.offset = get_docoffset(offset_path)
        self.docpath = os.path.join(ms_macro_path, "msmarco-docs.tsv")
        self.triplepath = os.path.join(ms_macro_path, "triples.tsv")
        self._init()

    def _combine_title_body(self, title, body):
        if self.title_body_mode == "both":
            return title + " " + body
        elif self.title_body_mode == "title":
            return title
        else:
            return body

    def _init(self):
        with open(self.triplepath, "rt") as f:
            for i, line in enumerate(tqdm(f, desc="Loading MS-Macro training data")):
                fields = line.rstrip().split("\t")
                [topicid, _, posdoc_id, negdoc_id] = fields
                self._index.append((topicid, posdoc_id, negdoc_id))

    def _get_doc(self, docid):
        with open(self.docpath) as f:
            _, _, title, body = getcontent(self.offset, docid, f)
            body = truncate_untokenized_doc(body)
            doc = self._combine_title_body(title, body)
            return doc

    def __getitem__(self, item):
        topicid, posdoc_id, negdoc_id = self._index[item]
        query = self.queries[topicid]
        posdoc = self._get_doc(posdoc_id)
        negdoc = self._get_doc(negdoc_id)
        return topicid, query, posdoc_id, posdoc, negdoc_id, negdoc

    def __len__(self):
        return len(self._index)

    class Collater():
        def __init__(self, bert_model: str = None):
            self.encoder = transformers.BertTokenizer.from_pretrained(bert_model or "bert-base-uncased")

        def __call__(self, examples):
            inputs = []
            labels = []
            topic_ids = []
            doc_ids = []

            for topic_id, query, posdoc_id, posdoc, negdoc_id, negdoc in examples:
                doc_ids.append(posdoc_id)
                topic_ids.append(topic_id)
                inputs.append((query, posdoc))
                labels.append(1)

                doc_ids.append(negdoc_id)
                topic_ids.append(topic_id)
                inputs.append((query, negdoc))
                labels.append(0)

            x = self.encoder.batch_encode_plus(inputs,
                                               return_tensors="pt",
                                               add_special_tokens=True,
                                               max_length=512,
                                               truncation_strategy="only_second")
            batch = topic_ids, doc_ids, x, torch.LongTensor(labels)
            return batch


class MsMacroDocReRankDataset(data.Dataset):
    def __init__(self, msmacro_path, mode="dev", title_body_mode="both", concat_query=False):
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
        self.concat_query = concat_query
        self._index = []
        self._init()

    def __len__(self):
        return len(self._index)

    def _combine_title_body(self, title, body):
        if self.title_body_mode == "both":
            return title + " " + body
        elif self.title_body_mode == "title":
            return title
        else:
            return body

    def _init(self):
        for topic_id, doc_ids in tqdm(self.top100.items(), desc="Loading Ms-Macro dev data"):
            query = self.queries[topic_id]
            for doc_id in doc_ids:
                if self.qrels:
                    label = 1 if doc_id in self.qrels[topic_id] else 0
                    self._index.append((topic_id, doc_id, query, label))
                else:
                    self._index.append((topic_id, doc_id, query))
                # if len(self._index)> 500:
                #     return

    def _get_doc(self, docid):
        with open(self.docs_path) as f:
            _, _, title, body = getcontent(self.docoffset, docid, f)
            body = truncate_untokenized_doc(body)
            doc = self._combine_title_body(title, body)
            return doc

    def __getitem__(self, item):
        fields = self._index[item]
        if len(fields) == 4:
            topic_id, doc_id, query, label = fields
        else:
            topic_id, doc_id, query = fields
            label = None

        doc = self._get_doc(doc_id)
        if self.concat_query and label is not None:
            if label == 0:
                doc = query + " " + doc
        return topic_id, doc_id, query, doc, label

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


def getcontent(docoffset, docid, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """

    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), \
        f"Looking for {docid}, found {line}"
    fields = line.rstrip().split("\t")
    if len(fields) > 4:
        docid, url, title = fields[:3]
        body = "\t".join(fields[3:])
        return docid, url, title, body
    elif len(fields) < 4:
        while len(fields) < 4:
            fields.append(" ")
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
                assert (len(rels[topicid]) <= 1)

    return rels


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


def truncate_untokenized_doc(body):
    return body[:5000]
