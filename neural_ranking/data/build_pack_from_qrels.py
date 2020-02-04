import numpy as np
import os
import shutil
from typing import List

anserini_root = os.environ.get("ANSERINI_HOME", ".")

# the following import order matters
from pyserini.setup import configure_classpath

configure_classpath(anserini_root)
from pyserini.search import pysearch

from functools import lru_cache
from pathlib import Path

import matchzoo as mz
import pandas as pd
import tqdm
from jnius import autoclass

from neural_ranking.data.topic_readers import TrecRobustTopicReader, \
    NtcirTopicReader, TopicReader, TrecXmlTopicReader

JString = autoclass('java.lang.String')
JIndexUtils = autoclass('io.anserini.index.IndexUtils')

TREC_PREL_COLUMNS = ["id_left", "id_right", "label", "algo", "prob"]
TREC_QREL_COLUMNS = ["id_left", "intent", "id_right", "label"]
NTCIR_QREL_COLUMNS = ["id_left", "id_right", "label"]


class DataBuilder():
    def __init__(self, index_path: Path,
                 topic_path: Path,
                 qrel_path: Path,
                 columns: List[str],
                 topic_reader: TopicReader,
                 searcher_name: str = "bm25"):

        self.qrel = qrel_path
        self.topic = topic_path
        index_path = str(index_path)
        self.index_utils = JIndexUtils(JString(index_path))
        self.searcher = pysearch.SimpleSearcher(index_path)
        self._set_searcher(searcher_name)
        self.columns = columns
        self.topic_reader = topic_reader

    def _set_searcher(self, name: str):
        if name == "bm25":
            self.searcher.set_bm25_similarity(0.9, 0.4)
        elif name == "bm25+rm3":
            self.searcher.set_bm25_similarity(0.9, 0.4)
            self.searcher.set_rm3_reranker(10, 10, 0.5)
        # TODO: implement more searchers

    @lru_cache(500000)
    def _extract_raw_doc(self, doc_id: str):
        # fetch raw document by id
        rawdoc = self.index_utils.getTransformedDocument(JString(doc_id))
        return rawdoc

    def extract_raw_docs(self, doc_ids: List[str], verbose: int = 0):
        docs = []
        ids = []
        for doc_id in tqdm.tqdm(doc_ids, disable=not verbose, desc="Extracting Raw Docs"):
            try:
                d = self._extract_raw_doc(doc_id)
                docs.append(d)
                ids.append(doc_id)
            except:
                pass

        return ids, docs

    def _parse_qrels(self):
        df = pd.read_csv(self.qrel, delimiter='\s+', names=self.columns)
        print("Qrels length: %d" % len(df))
        return df

    def _parse_topics(self):
        df = self.topic_reader(self.topic)
        print("Topics length: %d" % len(df))
        return df

    def build_datapack(self, output_path: Path, filter_topic_ids=None):
        topics_df = self._parse_topics()
        qrel_df = self._parse_qrels()
        df = qrel_df.join(topics_df, on="id_left", rsuffix="_topics", how="inner")[
            ["id_left", "text_left", "id_right", "label"]]

        if filter_topic_ids:
            filter_topic_ids = np.asarray(filter_topic_ids, dtype=df["id_left"].dtype)
            n_topics = len(set(df["id_left"]))
            df = df[~df["id_left"].isin(filter_topic_ids)]
            n_topics_filtered =  n_topics - len(set(df["id_left"]))
            print("Expected to filter %d topics, Filtered %d topics" % (len(filter_topic_ids), n_topics_filtered))

        qrel_len = len(df)
        distinct_doc_ids = list(set(df.id_right))
        retrieved_ids, retrieved_docs = self.extract_raw_docs(distinct_doc_ids,verbose=1)
        doc_df = pd.DataFrame({
            "text_right": retrieved_docs
        }, index=retrieved_ids)
        df = df.join(doc_df, on="id_right", how="inner")
        extracted_qrel_len = len(df)

        datapack = mz.data_pack.pack(df)
        datapack.save(output_path.joinpath("train"))
        print("Qrel Size: %d, Extracted %d documents" %(qrel_len, extracted_qrel_len))
        print("Datapack contain %d topics" % len(set(df["id_left"])))

        return datapack

    def search(self, query: str, hits=1000):
        hits = self.searcher.search(query, k=hits)
        doc_ids = [h.docid for h in hits]
        doc_ids, contents = self.extract_raw_docs(doc_ids, verbose=0)
        return doc_ids, contents

    def build_rerank_datapack(self, output_path: Path, hits=1000):
        topics_df = self._parse_topics()
        qrel_df = self._parse_qrels()

        rel_dict = {}
        for i in tqdm.tqdm(range(len(qrel_df)), desc="building rel dict"):
            row = qrel_df.iloc[i]
            rel_dict.setdefault(row.id_left, {})
            rel_dict[row.id_left][row.id_right] = row.label

        relation = []
        text_left = []
        text_right = []
        id_left = []
        id_right = []

        for i in tqdm.tqdm(range(len(topics_df)), desc="Building Rerank DataPack"):
            row = topics_df.iloc[i]
            doc_ids, doc_contents = self.search(row.text_left, hits=hits)

            text_left.extend([row.text_left for _ in doc_ids])
            id_left.extend([row.id_left for _ in doc_ids])

            id_right.extend(doc_ids)
            text_right.extend(doc_contents)

            for doc_id in doc_ids:
                r = rel_dict.get(row.id_left, {}).get(doc_id, 0)
                relation.append(r)

        pack = mz.data_pack.pack(pd.DataFrame(data={
            "label": relation,
            "text_right": text_right,
            "text_left": text_left,
            "id_left": id_left,
            "id_right": id_right
        }))
        pack.save(output_path.joinpath("rerank.%d" % hits))
        return pack


class TrecDataBuilder(DataBuilder):
    def __init__(self, index_path: Path, topic_path: Path, qrel_path: Path):
        super().__init__(index_path, topic_path, qrel_path, TREC_QREL_COLUMNS,
                         TrecRobustTopicReader())

class TrecXmlBuilder(DataBuilder):
    def __init__(self, index_path: Path, topic_path: Path, qrel_path: Path):
        super().__init__(index_path, topic_path, qrel_path, TREC_QREL_COLUMNS,
                         TrecXmlTopicReader())

class NtcirDataBuilder(DataBuilder):
    def __init__(self, index_path: Path, topic_path: Path, qrel_path: Path):
        super().__init__(index_path, topic_path, qrel_path, NTCIR_QREL_COLUMNS,
                         NtcirTopicReader())


def path(str):
    return Path(os.path.abspath((os.path.expanduser(str))))


def copy_qrel_and_topics(topics_path: Path, qrels_path: Path,
                         output_path: Path):
    shutil.copyfile(qrels_path, output_path.joinpath("qrels"))
    shutil.copy(topics_path, output_path.joinpath("topics"))

#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("--index", type=path)
#     parser.add_argument("--qrel", type=path)
#     parser.add_argument("--topic", type=path)
#     parser.add_argument("--topic-format", type=str, default='trec',
#                         choices=["trec", "ntcir", "xml"])
#     parser.add_argument("--output", type=path,
#                         default="./built_data/my-data-pack")
#     parser.add_argument("--filter", type=path)
#
#     parser.add_argument("--hits", type=int, default=1000)
#     args = parser.parse_args()
#
#     if args.topic_format == "trec":
#         builder = TrecDataBuilder(args.index, args.topic, args.qrel)
#     elif args.topic_format == "ntcir":
#         builder = NtcirDataBuilder(args.index, args.topic, args.qrel)
#     elif args.topic_format == "xml":
#         builder = TrecXmlBuilder(args.index, args.topic, args.qrel)
#     else:
#         raise ValueError
#
#     builder.build_datapack(args.output)
#     builder.build_rerank_datapack(args.output, args.hits)
#
#     copy_qrel_and_topics(args.topic, args.qrel, args.output)
