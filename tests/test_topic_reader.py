from unittest import TestCase

from dataset.topic_readers import TrecTopicReader


class TestTrecTopicReader(TestCase):

    def test_trec(self):
        reader = TrecTopicReader()

        df = reader("resources/topics_and_qrels/topics.robust04.301-450.601-700.txt")

        self.assertSequenceEqual([301, 302], df.index.tolist()[:2])
        self.assertSequenceEqual(["International Organized Crime", "Poliomyelitis and Post-Polio"],
                                 df["text_left"].tolist()[:2])
        self.assertSequenceEqual([699, 700], df.index.tolist()[-2:])
        self.assertSequenceEqual(["term limits", "gasoline tax U.S."],
                                 df["text_left"].tolist()[-2:])