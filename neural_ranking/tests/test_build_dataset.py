from unittest import TestCase

from dataset.build_pack_from_qrels import TrecDataBuilder


class TestTrecDataBuilder(TestCase):

    def test_trec(self):
        index = "/Users/zhaohaozeng/Documents/Anserini/lucene-index.robust04"
        topic = "resources/topics_and_qrels/topics.robust04.301-450.601-700.txt"
        qrel = "resources/topics_and_qrels/qrels.robust2004.txt"

        builder = TrecDataBuilder(index, topic, qrel)

        mypack = builder.build_datapack("./built_data/robust04.datapack")