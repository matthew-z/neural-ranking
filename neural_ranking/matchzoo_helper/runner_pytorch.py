import json
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import matchzoo as mz
from matchzoo.trainers import Trainer
from neural_ranking.utils.common import split_topics, \
    slice_datapack_by_left_ids, get_kfold_topics
import logging
PROJECT_FOLDER = Path("~/ir/neural_ranking/neural_ranking").expanduser()
DATA_FOLDER = Path("~/ir/neural_ranking/").expanduser().joinpath("built_data")
RESOURCES_FOLDER = PROJECT_FOLDER.joinpath("resources")


def dict_mean(inputs):
    result = {}
    for i, d in enumerate(inputs):
        for k, v in d.items():
            if i != 0 and k not in result:
                raise ValueError("Expect same keys in the dict list, "
                                 "but encountered different keys in inputs")

            result.setdefault(k, [])
            result[k].append(v)

    for k, v in result.items():
        result[k] = float(np.mean(v))

    return result


def folds_to_kfolds(folds):
    for test_id in range(len(folds)):
        train_fold = list(chain(folds[:test_id] + folds[test_id + 1:]))
        test_fold = folds[test_id]
        yield train_fold, test_fold


class ReRankTrainer(Trainer):
    def evaluate(
            self,
            dataloader):
        result = super().evaluate(dataloader)
        return result


class ReRankDataset(object):
    def __init__(self,
                 dataset="robust04",
                 topic_splits=None,
                 rerank_hits=1000):
        self.dataset = dataset
        self.dataset_path = DATA_FOLDER.joinpath(dataset)
        self.pack = mz.load_data_pack(self.dataset_path.joinpath("train"))
        self.rerank_pack = mz.load_data_pack(self.dataset_path.joinpath("rerank.%d" % rerank_hits))

        self.set_topic_splits(topic_splits)
        self.train_pack = slice_datapack_by_left_ids(self.pack, self.train_topics)

        self.rerank_pack_processed = None
        self.train_pack_processed = None

    def set_preprocessor(self, preprocessor):
        self.train_pack_processed = preprocessor.transform(
            self.train_pack)
        self.rerank_pack_processed = preprocessor.transform(
            self.rerank_pack)

    def set_topic_splits(self, topic_splits=None, dev_ratio=0.2, seed=None):

        if not topic_splits:
            topic_splits = split_topics(self.pack, seed=seed)

        if len(topic_splits) == 2:
            train_topics, self.test_topics = topic_splits
            self.train_topics, self.dev_topics = train_test_split(train_topics, test_size=dev_ratio, random_state=seed)
        elif len(topic_splits) == 3:
            self.train_topics, self.dev_topics, self.test_topics = topic_splits
        else:
            raise ValueError("Expect to get two or three splits, got %d" % len(topic_splits))

        self.train_pack = slice_datapack_by_left_ids(self.pack, self.train_topics)

    @property
    def dev_pack_processed(self):
        return slice_datapack_by_left_ids(self.rerank_pack_processed, self.dev_topics)

    @property
    def test_pack_processed(self):
        return slice_datapack_by_left_ids(self.rerank_pack_processed, self.test_topics)

    def _get_predefined_kfold_splits(self):
        if self.dataset == "robust04":
            filepath = RESOURCES_FOLDER.joinpath("fine_tuning").joinpath("robust04-5folds.json")
            folds = json.load(filepath.open())
            return list(folds_to_kfolds(folds))

    def get_kfold_splits(self, k=5, seed=None):
        predefined_kfold_splits = self._get_predefined_kfold_splits()
        return predefined_kfold_splits or get_kfold_topics(self.pack, k=k, seed=seed)


class Runner(object):
    def __init__(self, dataset: ReRankDataset, embedding=None, log_path="log"):
        self.dataset = dataset

        if embedding is None:
            self.raw_embedding = mz.embedding.GloVe(dim=50, name="6B")
        else:
            self.raw_embedding = embedding

        self.log_path = Path(log_path).absolute()
        self.logger = logging.getLogger("Runner")

    def prepare(self, model_class, task=None, topic_split=None, preprocessor=None):
        self.dataset.set_topic_splits(topic_split)
        self.model_class = model_class

        preprocessor = preprocessor or model_class.get_default_preprocessor(
            truncated_length_left=24,
            truncated_length_right=256,
            multiprocessing=True
        )

        (self.model,
         self.preprocessor,
         self.dataset_builder,
         self.dataloader_builder) = mz.auto.prepare(
            task=task or get_ranking_task(),
            model_class=model_class,
            data_pack=self.dataset.train_pack,
            embedding=self.raw_embedding,
            preprocessor=preprocessor
        )
        self.dataset.set_preprocessor(self.preprocessor)

    def run_kfold(self, kfold_topic_splits=None, **kwargs):
        results = []
        kfolds = kfold_topic_splits or self.dataset.get_kfold_splits()
        for i, topic_splits in enumerate(kfolds):
            self.logger.info("Fold - %d of %d" % (i, len(kfolds)))
            self.dataset.set_topic_splits(topic_splits)
            results.append(self.run(**kwargs))
        result =  dict_mean(results)
        self.logger.info(result)
        return result


    def get_default_configs(self):
        return {
            "epochs": 10,
            "patience": 5,
            "train_ratio": 1.0,
            "optimizer_cls": torch.optim.Adam,
            "lr": None,
            "batch_size": 128
        }

    def run(self, configs=None):
        if configs is None:
            configs = self.get_default_configs()

        train_ratio = configs.get("train_ratio", 1.0)
        if train_ratio < 1.0:
            train_size = int(len(self.dataset.train_pack_processed) * train_ratio)
            training_pack = self.dataset.train_pack_processed[:train_size]
        else:
            training_pack = self.dataset.train_pack_processed

        # Setup data
        batch_size = configs.get("batch_size", 128)
        trainset = self.dataset_builder.build(
            training_pack,
            batch_size=batch_size,
            sort=False
        )

        train_loader = self.dataloader_builder.build(trainset)
        eval_dataset_builder = mz.dataloader.DatasetBuilder(
            batch_size=batch_size,
            shuffle=False,
            sort=False,
            resample=False,
            mode="point"
        )

        dev_dataset = eval_dataset_builder.build(self.dataset.dev_pack_processed)
        dev_loader = self.dataloader_builder.build(dataset=dev_dataset,
                                                   stage="dev")

        # Training setting
        optimizer = configs.get("optimizer_cls", torch.optim.Adam)(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)

        trainer = ReRankTrainer(
            model=self.model,
            optimizer=optimizer,
            trainloader=train_loader,
            validloader=dev_loader,
            scheduler=scheduler,
            epochs=configs.get("epochs", 10),
            patience=configs.get("patience", 5),
        )

        trainer.run()
        # Restore the best model according to the dev scores.
        trainer.restore_model(trainer._save_dir.joinpath('model.pt'))

        # Evaluate Model on the test data
        test_set = eval_dataset_builder.build(self.dataset.test_pack_processed)
        test_loader = self.dataloader_builder.build(dataset=test_set,
                                                    stage="dev")
        test_score = trainer.evaluate(test_loader)
        return test_score

def get_ranking_task(loss=None):
    ranking_task = mz.tasks.Ranking(losses=loss or mz.losses.RankHingeLoss())
    ranking_task.metrics = [
        mz.metrics.MeanAveragePrecision(),
        mz.metrics.Precision(k=30),
    ]
    return ranking_task
