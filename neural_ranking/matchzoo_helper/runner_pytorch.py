from pathlib import Path

import dill
import numpy as np
import torch

import matchzoo as mz
from matchzoo.trainers import Trainer
from neural_ranking.utils.common import split_topics, \
    slice_datapack_by_left_ids

DATA_FOLDER = Path("~/ir/neural_ranking/built_data/").expanduser()


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


class ReRankTrainer(Trainer):
    def evaluate(
            self,
            dataloader):
        result = super().evaluate(dataloader)
        return result


class Runner(object):
    def __init__(self, dataset="robust04", embedding=None, log_path="log"):
        self.dataset = dataset
        self.prepare_data(DATA_FOLDER.joinpath(dataset))

        if embedding is None:
            self.raw_embedding = mz.embedding.GloVe(dim=50, name="6B")
        else:
            self.raw_embedding = embedding

        self.log_path = Path(log_path).absolute()

    def prepare_data(self, datapath: Path, rerank_hits=1000):
        pack = mz.load_data_pack(datapath.joinpath("train")).shuffle()
        self.rerank_pack = mz.load_data_pack(
            datapath.joinpath("rerank.%d" % rerank_hits))
        self.train_topics, self.dev_topics, self.test_topics = split_topics(
            pack)
        self.train_pack = slice_datapack_by_left_ids(pack, self.train_topics)

    def prepare_model(self, model_class, train_ratio=1.0, task=None):
        self.train_ratio = train_ratio
        train_size = int(len(self.train_pack) * train_ratio)
        truncated_train_pack = self.train_pack[:train_size]

        self.task = task or get_ranking_task()
        self.model_class = model_class

        (self.model,
         self.preprocessor,
         self.dataset_builder,
         self.dataloader_builder) = mz.auto.prepare(
            task=self.task,
            model_class=model_class,
            data_pack=truncated_train_pack,
            embedding=self.raw_embedding,
            preprocessor=model_class.get_default_preprocessor(
                truncated_length_left=24,
                truncated_length_right=256,
                multiprocessing=True
            )
        )

        self.train_pack_processed = self.preprocessor.transform(
            truncated_train_pack)
        self.rerank_pack_processed = self.preprocessor.transform(
            self.rerank_pack)
        self.split_dev_test()

    def split_dev_test(self):
        self.dev_pack_processed = slice_datapack_by_left_ids(
            self.rerank_pack_processed, self.dev_topics)
        self.test_pack_processed = slice_datapack_by_left_ids(
            self.rerank_pack_processed, self.test_topics)

    def run(self, epochs=1):
        # Setup data
        trainset = self.dataset_builder.build(
            self.train_pack_processed,
            batch_size=256,
            sort=False
        )

        train_loader = self.dataloader_builder.build(trainset)
        eval_dataset_builder = mz.dataloader.DatasetBuilder(
            batch_size=512,
            shuffle=False,
            sort=False,
            resample=False,
            mode="point"
        )

        dev_dataset = eval_dataset_builder.build(self.dev_pack_processed)
        dev_loader = self.dataloader_builder.build(dataset=dev_dataset,
                                                   stage="dev")

        # Training setting
        optimizer = torch.optim.Adadelta(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)

        trainer = ReRankTrainer(
            model=self.model,
            optimizer=optimizer,
            trainloader=train_loader,
            validloader=dev_loader,
            scheduler=scheduler,
            epochs=epochs,
            patience=3,
        )

        trainer.run()
        # Restore the best model according to the dev scores.
        trainer.restore_model(trainer._save_dir.joinpath('model.pt'))


        # Evaluate Model on the test data
        test_set = eval_dataset_builder.build(self.test_pack_processed)
        test_loader = self.dataloader_builder.build(dataset=test_set,
                                                    stage="dev")

        test_preds = trainer.predict(test_loader)
        topic_ids = self.test_pack_processed.frame().id_left
        document_ids = self.test_pack_processed.frame().id_right
        labels = self.test_pack_processed.frame().label
        test_score = trainer.evaluate(test_loader)
        to_check = [test_preds, topic_ids, document_ids, labels,
                    test_loader.label, test_score, test_loader, test_set]
        print("Evaluation Result: %s" % test_score)
        dill.dump(to_check, open("/tmp/to_check.dill", "wb"))


def get_ranking_task(loss=None):
    ranking_task = mz.tasks.Ranking(losses=loss or mz.losses.RankHingeLoss())
    ranking_task.metrics = [
        mz.metrics.MeanAveragePrecision(),
        mz.metrics.Precision(k=30),
    ]
    return ranking_task
