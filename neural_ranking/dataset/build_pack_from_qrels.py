import os
import sys

anserini_root = os.environ.get("ANSERINI_HOME", ".")
sys.path += [os.path.join(anserini_root, 'src/main/python')]

from pyserini.setup import configure_classpath
from pyserini.search import pysearch

configure_classpath(anserini_root)
from jnius import autoclass
import pandas as pd
from neural_ranking.dataset.topic_readers import TrecTopicReader, NtcirTopicReader
from neural_ranking.dataset.web_extractor import Robust04Extractor, SimpleHTMLExtractor
import matchzoo as mz
from pathlib import Path
from functools import lru_cache
import tqdm

JString = autoclass('java.lang.String')
JIndexUtils = autoclass('io.anserini.index.IndexUtils')

TREC_QREL_COLUMNS = ["id_left", "intent", "id_right", "label"]
NTCIR_QREL_COLUMNS = ["id_left", "id_right", "label"]


def get_extractor_cls(name):
    if name=="robust04":
        return Robust04Extractor
    return SimpleHTMLExtractor


class DataBuilder():
    def __init__(self, index_path, topic_path, qrel_path, columns, topic_reader, searcher_name="bm25"):
        self.qrel = qrel_path
        self.topic = topic_path
        self.index_utils = JIndexUtils(JString(index_path))
        self.searcher = pysearch.SimpleSearcher(index_path)
        self._set_searcher(searcher_name)
        self.columns = columns
        self.topic_reader = topic_reader

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

    def extract_raw_docs(self, doc_ids):
        docs = []
        ids = []
        for doc_id in tqdm.tqdm(doc_ids):
            try:
                d = self._extract_raw_doc(doc_id)
                docs.append(d)
                ids.append(doc_id)
            except:
                pass

        return ids,docs


    def _parse_qrels(self):
        df =  pd.read_csv(self.qrel, delimiter=" ", names=self.columns)
        print("Qrels length: %d" % len(df))
        return df

    def _parse_topics(self):
        df =  self.topic_reader(self.topic)
        print("Topics length: %d" % len(df))
        return df

    def build_datapack(self, output_path):
        topics_df = self._parse_topics()
        qrel_df = self._parse_qrels()
        df = qrel_df.join(topics_df, on="id_left",rsuffix="_topics", how="inner")[["id_left", "text_left", "id_right", "label"]]

        distinct_doc_ids = list(set(df.id_right))
        retrieved_ids,retrieved_docs  = self.extract_raw_docs(distinct_doc_ids)

        doc_df = pd.DataFrame({
            "text_right": retrieved_docs
        }, index=retrieved_ids)

        df = df.join(doc_df, on="id_right", how="inner")

        datapack = mz.data_pack.pack(df)
        datapack.save(output_path)

        return datapack

    def search(self, query):
        hits = self.searcher.search(query)
        doc_ids = [h.docid for h in hits]
        contents, doc_ids = self.extract_raw_docs(doc_ids)
        return doc_ids, contents


    def build_rerank_datapack(self, output_path: Path):
        output_path = output_path.with_suffix(".rerank")
        topics_df = self._parse_topics()
        qrel_df = self._parse_qrels()

        for i in range(len(topics_df)):
            row = topics_df.iloc[i]
            doc_ids, doc_contents = self.search(row.text_left)

            left = [
                [row.id_left, row.text_left]
            ]

            right = [
                []
            ]



class TrecDataBuilder(DataBuilder):
    def __init__(self, index_path, topic_path, qrel_path):
        super().__init__(index_path, topic_path, qrel_path, TREC_QREL_COLUMNS, TrecTopicReader())


class NtcirDataBuilder(DataBuilder):
    def __init__(self, index_path, topic_path, qrel_path):
        super().__init__(index_path, topic_path, qrel_path, NTCIR_QREL_COLUMNS, NtcirTopicReader())


def path(str):
    return os.path.abspath((os.path.expanduser(str)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--index", type=path)
    parser.add_argument("--qrel", type=path)
    parser.add_argument("--topic", type=path)
    parser.add_argument("--format", type=str, default='trec')
    parser.add_argument("--output", type=path, default="./built_data/my-data-pack")
    args = parser.parse_args()

    if args.format == "trec":
        TrecDataBuilder(args.index, args.topic, args.qrel).build_datapack(args.output)
    if args.format == "ntcir":
        NtcirDataBuilder(args.index, args.topic, args.qrel).build_datapack(args.output)