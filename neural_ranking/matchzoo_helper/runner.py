import os
from pathlib import Path

import numpy as np
import torch
from comet_ml import Experiment

import matchzoo as mz
from neural_ranking.dataset.asr.asr_collection import AsrCollection
from neural_ranking.evaluation.robustness import get_robustness_metrics
from neural_ranking.matchzoo_helper.dataset import ReRankDataset
from neural_ranking.matchzoo_helper.utils import dict_mean, ReRankTrainer, get_ranking_task


class Runner(object):
    def __init__(self,
                 dataset: ReRankDataset,
                 preprocessor_path=None,
                 embedding=None,
                 log_path: str = "log",
                 checkpoint_path="checkpoint",
                 fp16=False):
        self.dataset = dataset
        self.raw_embedding = embedding
        self.preprocessor = None
        self.name = None
        self.fp16 = fp16

        self.checkpoint_path = Path(checkpoint_path).absolute()
        self.log_path = Path(log_path).absolute()
        self.preprocessor_path = preprocessor_path

    def _log_hparams(self, configs):
        if self.logger:
            self.logger.log_parameters(configs)

    def prepare(self, model_class, task=None, preprocessor=None,
                force_update_preprocessor=True, config=None, extra_terms=None, experiment: Experiment = None):
        self.model_class = model_class
        self.logger = experiment

        if self.preprocessor is None:
            if self.preprocessor_path is not None and os.path.exists(self.preprocessor_path):
                preprocessor = mz.load_preprocessor(self.preprocessor_path)
                self.preprocessor = preprocessor
            else:
                preprocessor = preprocessor or self.model_class.get_default_preprocessor(truncated_length_left=20,
                                                                                         truncated_length_right=1024,
                                                                                         truncated_mode="post")
                preprocessor.multiprocessing = 0 if self.dataset.debug_mode else 1
                preprocessor.extra_terms = extra_terms

        update_preprocessor = force_update_preprocessor or type(self.preprocessor) != type(preprocessor)
        if not update_preprocessor and self.preprocessor:
            preprocessor = self.preprocessor
            preprocessor.fit = lambda self, x: None

        (self.model,
         self.preprocessor,
         self.dataset_builder,
         self.dataloader_builder) = mz.auto.prepare(
            task=task or get_ranking_task(),
            model_class=model_class,
            data_pack=self.dataset.pack,
            embedding=self.raw_embedding,
            preprocessor=preprocessor,
            config=config
        )

        if update_preprocessor:
            self.dataset.set_preprocessor(self.preprocessor)


    def _reset_model(self, configs=None):
        self.model._params.update(configs)
        assert self.model._params["embedding"] is not None
        self.model.build()
        return self.model

    def train(self, optimizer=None, optimizer_fn=None, scheduler=None,
              scheduler_fn=None, run_name=None, save_dir=None,
              train=True, devices=None, **kwargs):

        configs = self._get_default_configs()
        configs.update(kwargs)
        self._reset_model(configs)
        self._log_hparams(configs)
        self._log_hparams(self.model._params.to_dict())

        run_name = run_name or self._get_default_run_name(configs)
        train_loader, dev_loader = self.get_dataloaders(configs)

        if optimizer_fn:
            optimizer = optimizer_fn(self.model)
        else:
            optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=configs["lr"])

        save_dir = save_dir or self.checkpoint_path.joinpath(run_name)
        os.makedirs(save_dir, exist_ok=True)

        self.trainer = ReRankTrainer(
            model=self.model,
            optimizer=optimizer,
            trainloader=train_loader,
            validloader=dev_loader,
            scheduler=scheduler,
            scheduler_fn=scheduler_fn,
            epochs=configs.get("epochs"),
            patience=configs.get("patience"),
            device=devices or "cuda",
            save_dir=save_dir,
            fp16=self.fp16
        )

        if train:
            self.trainer.run()

        # Restore the best model according to the dev scores.
        self.trainer.restore_model(save_dir.joinpath('model.pt'))
        dev_score = self.trainer.evaluate(dev_loader)
        if self.logger:
            self.logger.log_metrics({str(metric):score for metric, score in dev_score.items()}, prefix="dev__")

    def predict(self, data_pack, batch_size=32):
        eval_dataset_builder = mz.dataloader.DatasetBuilder(
            batch_size=batch_size,
            shuffle=False,
            sort=False,
            resample=False,
            mode="point"
        )
        dataset = eval_dataset_builder.build(self.preprocessor.transform(data_pack))
        loader = self.dataloader_builder.build(dataset=dataset, stage="test")
        preds = self.trainer.predict(loader)
        pred_dict = {}
        for pred, docid in zip(preds, loader.id_right):
            pred_dict[docid] = float(pred)
        return pred_dict

    def eval_asrc(self, asr_collection: AsrCollection):
        pred_dict = self.predict(asr_collection.data_pack)
        rounds_dict = {}
        for docid, score in pred_dict.items():
            _, round, query_id, author_id = docid.split("-")
            query_id = int(query_id)
            round = int(round)
            author_id = int(author_id)
            rounds_dict.setdefault(round, {})
            rounds_dict[round].setdefault(query_id, [])
            rounds_dict[round][query_id].append((score, author_id))
        result = {}
        for metric_name, metric_fn in get_robustness_metrics().items():
            metric_score = []
            for round in range(2, 9):
                queries_curr = rounds_dict[round]
                queries_prev = rounds_dict[round - 1]

                scores_per_round = []
                for query in queries_curr:
                    docs_curr = sorted(queries_curr[query], reverse=True)
                    docs_prev = sorted(queries_prev[query], reverse=True)
                    score = metric_fn(docs_prev, docs_curr)
                    scores_per_round.append(score)

                metric_score.append(np.mean(scores_per_round))

            result[metric_name] = np.mean(metric_score)
            if self.logger:
                self.logger.log_metrics(result, prefix="test_")
        return result

    def train_kfold(self, kfold_topic_splits=None, fold_num=None, **kwargs):
        results = []
        kfolds = kfold_topic_splits or self.dataset.get_kfold_splits()
        for i, topic_splits in enumerate(kfolds):
            if fold_num and i >= fold_num:
                break
            print(">>>> Fold - %d of %d" % (i, len(kfolds)))
            self.dataset.init_topic_splits(topic_splits)
            results.append(self.train(**kwargs))
        result = dict_mean(results)
        return result

    def _get_default_configs(self):
        return {
            "epochs": 10,
            "patience": 5,
            "train_ratio": 1.0,
            "optimizer_cls": torch.optim.Adam,
            "lr": 1e-3,
            "batch_size": 64,
        }

    def _get_default_run_name(self, configs):
        name = [self.model_class.__name__]
        return ".".join(name)

    def get_dataloaders(self, configs):
        train_ratio = configs.get("train_ratio")
        if train_ratio < 1.0:
            train_size = int(len(self.dataset.train_pack_processed) * train_ratio)
            training_pack = self.dataset.train_pack_processed[:train_size]
        else:
            training_pack = self.dataset.train_pack_processed
        # Setup data
        batch_size = configs.get("batch_size")
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

        return train_loader, dev_loader
