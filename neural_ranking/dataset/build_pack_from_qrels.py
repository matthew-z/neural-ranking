import os
import sys

anserini_root = os.environ.get("ANSERINI_HOME", ".")
sys.path += [os.path.join(anserini_root, 'src/main/python')]

# the following import order matters
from pyserini.setup import configure_classpath

configure_classpath(anserini_root)
from pyserini.search import pysearch

from functools import lru_cache
from pathlib import Path

import dill
import matchzoo as mz
import pandas as pd
import tqdm
from jnius import autoclass

from neural_ranking.dataset.topic_readers import TrecTopicReader, NtcirTopicReader

JString = autoclass('java.lang.String')
JIndexUtils = autoclass('io.anserini.index.IndexUtils')

TREC_QREL_COLUMNS = ["id_left", "intent", "id_right", "label"]
NTCIR_QREL_COLUMNS = ["id_left", "id_right", "label"]


class DataBuilder():
    def __init__(self, index_path, topic_path, qrel_path, columns, topic_reader, searcher_name="bm25", hits=1000):
        self.qrel = qrel_path
        self.topic = topic_path
        self.index_utils = JIndexUtils(JString(index_path))
        self.searcher = pysearch.SimpleSearcher(index_path)
        self._set_searcher(searcher_name)
        self.columns = columns
        self.topic_reader = topic_reader
        self.hits = hits

    def _set_searcher(self, name):
        if name == "bm25":
            self.searcher.set_bm25_similarity(0.9, 0.4)
        elif name == "bm25+rm3":
            self.searcher.set_bm25_similarity(0.9, 0.4)
            self.searcher.set_rm3_reranker(10, 10, 0.5)
        # TODO: implement more searchers

    @lru_cache(500000)
    def _extract_raw_doc(self, doc_id):
        # fetch raw document by id
        rawdoc = self.index_utils.getTransformedDocument(JString(doc_id))
        return rawdoc

    def extract_raw_docs(self, doc_ids, verbose=0):
        docs = []
        ids = []
        for doc_id in tqdm.tqdm(doc_ids, disable=not verbose):
            try:
                d = self._extract_raw_doc(doc_id)
                docs.append(d)
                ids.append(doc_id)
            except:
                pass

        return ids, docs

    def _parse_qrels(self):
        df = pd.read_csv(self.qrel, delimiter=" ", names=self.columns)
        print("Qrels length: %d" % len(df))
        return df

    def _parse_topics(self):
        df = self.topic_reader(self.topic)
        print(df.columns)
        print("Topics length: %d" % len(df))
        return df

    def build_datapack(self, output_path):
        topics_df = self._parse_topics()
        qrel_df = self._parse_qrels()
        df = qrel_df.join(topics_df, on="id_left", rsuffix="_topics", how="inner")[
            ["id_left", "text_left", "id_right", "label"]]

        distinct_doc_ids = list(set(df.id_right))
        retrieved_ids, retrieved_docs = self.extract_raw_docs(distinct_doc_ids, verbose=1)

        doc_df = pd.DataFrame({
            "text_right": retrieved_docs
        }, index=retrieved_ids)

        df = df.join(doc_df, on="id_right", how="inner")

        datapack = mz.data_pack.pack(df)
        datapack.save(output_path)

        return datapack

    def search(self, query):
        hits = self.searcher.search(query, k=self.hits)
        doc_ids = [h.docid for h in hits]
        doc_ids, contents = self.extract_raw_docs(doc_ids, verbose=0)
        return doc_ids, contents

    def build_rerank_datapack(self, output_path: str):
        topics_df = self._parse_topics()
        qrel_df = self._parse_qrels()

        rel_dict = {}
        for i in tqdm.tqdm(range(len(qrel_df)), desc="building rel dict"):
            row = qrel_df.iloc[i]
            rel_dict.setdefault(row.id_left, {})
            rel_dict[row.id_left][row.id_right] = row.label
        packs = []

        for i in tqdm.tqdm(range(len(topics_df))):
            row = topics_df.iloc[i]
            doc_ids, doc_contents = self.search(row.text_left)

            text_left = [row.text_left for _ in doc_ids]
            id_left = [row.id_left for _ in doc_ids]
            relation = []

            for doc_id in doc_ids:
                r = rel_dict.get(row.id_left, {}).get(doc_id, 0)
                relation.append(r)

            pack = mz.data_pack.pack(pd.DataFrame(data={
                "label": relation,
                "text_right": doc_contents,
                "text_left": text_left,
                "id_left": id_left,
                "id_right": doc_ids

            }))

            assert len(pack) !=0
            packs.append(pack)

        save(packs, output_path, data_filename="rerank.dill")
        return packs


class TrecDataBuilder(DataBuilder):
    def __init__(self, index_path, topic_path, qrel_path):
        super().__init__(index_path, topic_path, qrel_path, TREC_QREL_COLUMNS, TrecTopicReader())


class NtcirDataBuilder(DataBuilder):
    def __init__(self, index_path, topic_path, qrel_path):
        super().__init__(index_path, topic_path, qrel_path, NTCIR_QREL_COLUMNS, NtcirTopicReader())


def path(str):
    return os.path.abspath((os.path.expanduser(str)))


def save(obj, dirpath, data_filename="data.dill"):
    dirpath = Path(dirpath)
    data_file_path = dirpath.joinpath(data_filename)

    if data_file_path.exists():
        raise FileExistsError(
            f'{data_file_path} already exist, fail to save')
    elif not dirpath.exists():
        dirpath.mkdir()

    dill.dump(obj, open(data_file_path, mode='wb'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--index", type=path)
    parser.add_argument("--qrel", type=path)
    parser.add_argument("--topic", type=path)
    parser.add_argument("--format", type=str, default='trec', choices=["trec", "ntcir"])
    parser.add_argument("--output", type=path, default="./built_data/my-data-pack")
    parser.add_argument("--hits", type=int, default=1000)
    args = parser.parse_args()

    if args.format == "trec":
        builder = TrecDataBuilder(args.index, args.topic, args.qrel)
    elif args.format == "ntcir":
        builder = NtcirDataBuilder(args.index, args.topic, args.qrel)
    else:
        raise ValueError

    # builder.build_datapack(args.output)
    builder.build_rerank_datapack(args.output)
