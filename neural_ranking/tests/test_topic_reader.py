from unittest import TestCase

from neural_ranking.data.topic_readers import *


class TestTrecTopicReader(TestCase):

    def test_robust04(self):
        reader = TrecRobustTopicReader()

        df = reader("../resources/topics_and_qrels/topics.robust04.301-450.601-700.txt")

        self.assertSequenceEqual([301, 302], df.index.tolist()[:2])
        self.assertSequenceEqual(["International Organized Crime", "Poliomyelitis and Post-Polio"],
                                 df["text_left"].tolist()[:2])
        self.assertSequenceEqual([699, 700], df.index.tolist()[-2:])
        self.assertSequenceEqual(["term limits", "gasoline tax U.S."],
                                 df["text_left"].tolist()[-2:])


    def test_web(self):
        reader = TrecXmlTopicReader()

        df = reader("../resources/topics_and_qrels/topics.web.1-300.txt")

        self.assertSequenceEqual([1, 2], df.index.tolist()[:2])
        self.assertSequenceEqual(["obama family tree", "french lick resort and casino"],
                                 df["text_left"].tolist()[:2])
        self.assertSequenceEqual([299, 300], df.index.tolist()[-2:])
        self.assertSequenceEqual(["pink slime in ground beef", "how to find the mean"],
                                 df["text_left"].tolist()[-2:])