import json
from pathlib import Path

from sklearn.model_selection import train_test_split

import matchzoo as mz
from neural_ranking.runners.utils import folds_to_kfolds
from neural_ranking.utils.common import split_topics, slice_datapack_by_left_ids, get_kfold_topics

PROJECT_FOLDER = Path("~/ir/neural_ranking/neural_ranking").expanduser()
DATA_FOLDER = Path("~/ir/neural_ranking/").expanduser().joinpath("built_data")
RESOURCES_FOLDER = PROJECT_FOLDER.joinpath("resources")


class ReRankDataset(object):
    def __init__(self,
                 dataset="robust04",
                 rerank_hits=1000,
                 shuffle=False,
                 debug_mode=False):
        self.dataset = dataset
        self.dataset_path = DATA_FOLDER.joinpath(dataset)
        self.pack = mz.load_data_pack(self.dataset_path.joinpath("train"))
        self.rerank_pack = mz.load_data_pack(self.dataset_path.joinpath("rerank.%d" % rerank_hits))
        self.topic_path = self.dataset_path.joinpath("topics")
        self.qrel_path = self.dataset_path.joinpath("qrels")
        self.debug_mode = debug_mode
        if shuffle:
            self.pack = self.pack.shuffle()
            self.rerank_pack = self.rerank_pack.shuffle()
        if debug_mode:
            self.pack = self.pack.shuffle()[:1000]
            self.rerank_pack = self.rerank_pack.shuffle()[:500]

        self.rerank_pack_processed = None
        self.pack_processed = None

    def set_preprocessor(self, preprocessor):
        self.pack_processed = preprocessor.transform(
            self.pack)
        self.rerank_pack_processed = preprocessor.transform(
            self.rerank_pack)

    def init_topic_splits(self, topic_splits=None, dev_ratio=0.2, test_ratio=0.2, seed=None):

        if not topic_splits:
            topic_splits = split_topics(self.pack, valid_size=dev_ratio, test_size=test_ratio, seed=seed)

        if len(topic_splits) == 2:
            if test_ratio > 0:
                train_topics, self.test_topics = topic_splits
                self.train_topics, self.dev_topics = train_test_split(train_topics,
                                                                      test_size=dev_ratio, random_state=seed)
            else:
                self.train_topics, self.dev_topics = topic_splits
                self.test_topics = []
        elif len(topic_splits) == 3:
            self.train_topics, self.dev_topics, self.test_topics = topic_splits
        else:
            raise ValueError("Expect to get two or three splits, got %d" % len(topic_splits))

    @property
    def train_pack_processed(self):
        return slice_datapack_by_left_ids(self.pack_processed, self.train_topics)

    @property
    def dev_pack_processed(self):
        return slice_datapack_by_left_ids(self.rerank_pack_processed, self.dev_topics)

    @property
    def test_pack_processed(self):
        return slice_datapack_by_left_ids(self.rerank_pack_processed, self.test_topics)

    @property
    def test_pack(self):
        return slice_datapack_by_left_ids(self.rerank_pack, self.test_topics)

    @property
    def train_pack_positive_num(self, threshold=0):
        return sum(self.train_pack_processed.relation.label > threshold)


class KFoldRerankDataset(ReRankDataset):
    def set_kfold_splits(self, fold_index=0, k=5, seed=None):
        fold = self.get_kfold_splits(k, seed=seed)[fold_index]
        self.init_topic_splits(fold, seed=seed)

    def _get_predefined_kfold_splits(self):
        if self.dataset == "robust04":
            filepath = RESOURCES_FOLDER.joinpath("fine_tuning").joinpath("robust04-5folds.json")
            folds = [list(map(int, fold)) for fold in json.load(filepath.open())]
            return list(folds_to_kfolds(folds))

    def get_kfold_splits(self, k=5, seed=None):
        predefined_kfold_splits = self._get_predefined_kfold_splits()
        return predefined_kfold_splits or get_kfold_topics(self.pack, k=k, seed=seed)
