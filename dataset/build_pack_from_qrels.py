import os
import sys

anserini_root = os.environ.get("ANSERINI_HOME", ".")
sys.path += [os.path.join(anserini_root, 'src/main/python')]

from pyserini.setup import configure_classpath
configure_classpath(anserini_root)
from jnius import autoclass
import pandas as pd
from dataset.topic_readers import TopicReader, TrecTopicReader
from dataset.web_extractor import Robust04Extractor, SimpleHTMLExtractor
import matchzoo as mz
from functools import lru_cache

JString = autoclass('java.lang.String')
JIndexUtils = autoclass('io.anserini.index.IndexUtils')

TREC_QREL_COLUMNS = ["id_left", "intent", "id_right", "label"]

def get_extractor_cls(name):
    if name=="robust04":
        return Robust04Extractor
    return SimpleHTMLExtractor


class TrecDataBuilder():
    def __init__(self, index_path, topic_path, qrel_path):
        self.qrel = qrel_path
        self.topic = topic_path
        self.index_utils = JIndexUtils(JString(index_path))
        self.extractor = Robust04Extractor()

    @lru_cache(500000)
    def _extract_raw_doc(self, doc_id):
        # fetch raw document by id
        rawdoc = self.index_utils.getRawDocument(JString(doc_id))
        return rawdoc

    def extract_raw_docs(self, doc_ids):
        return [self._extract_raw_doc(doc_id) for doc_id in doc_ids]


    def _parse_qrels(self):
        df =  pd.read_csv(self.qrel, delimiter=" ", names=TREC_QREL_COLUMNS)
        return df

    def _parse_topics(self):
        reader = TrecTopicReader()
        return reader(self.topic)

    def build_datapack(self, output_path):
        topics_df = self._parse_topics()
        qrel_df = self._parse_qrels()
        df = qrel_df.join(topics_df, on="id_left",rsuffix="_topics", how="inner")[["id_left", "text_left", "id_right", "label"]]


        distinct_doc_ids = list(set(df.id_right))
        distinct_raw_docs = self.extract_raw_docs(distinct_doc_ids)
        distinct_docs = self.extractor.bulk_extract(distinct_raw_docs)

        doc_df = pd.DataFrame({
            "text_right": distinct_docs
        }, index=distinct_doc_ids)

        df = df.join(doc_df, on="id_right", how="inner")

        datapack = mz.data_pack.pack(df)
        datapack.save(output_path)

        return datapack

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