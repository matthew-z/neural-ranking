import gc
import os
from pathlib import Path

import numpy as np
import torch
from comet_ml import Experiment

import matchzoo as mz
from neural_ranking.data.callbacks import InsertQueryToDoc
from neural_ranking.dataset.asr.asr_collection import AsrCollection
from neural_ranking.evaluation import robustness
from neural_ranking.matchzoo_helper.dataset import ReRankDataset
from neural_ranking.matchzoo_helper.utils import dict_mean, ReRankTrainer, get_ranking_task


def calculate_model_norm(model):
    no_decay = ['bias', 'LayerNorm.weight']
    total_norm = 0
    embedding_norm = 0
    for name, param in model.named_parameters():

        if any(nd in name for nd in no_decay):
            continue
        elif "embedding" in name:
            embedding_norm += (param.data.norm(2)) ** 2
        else:
            total_norm += (param.data.norm(2)) ** 2

    total_norm = (total_norm ** (1 / 2)).item()
    embedding_norm = (embedding_norm ** (1 / 2)).item()
    return total_norm, embedding_norm




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

    def _load_basic_preprocessor(self, extra_terms):
        dataset_name = self.dataset.dataset
        preprocessor_name = "basic" if self.model_class != mz.models.Bert else "bert"
        preprocessor_folder = self.preprocessor_path
        save_name = ".".join([preprocessor_name, dataset_name])
        save_path = os.path.join(preprocessor_folder, save_name)
        if os.path.exists(save_path) and self.model_class != mz.models.Bert:
            preprocessor = mz.load_preprocessor(save_path)
            print("Load Preprocessor from %s" % save_path)
            preprocessor.fit = lambda *args, **argv: None
            save_path = None
        else:
            print("Init Preprocessor" )
            preprocessor = self.model_class.get_default_preprocessor(
                truncated_length_left=20,
                truncated_length_right=1024 if self.model_class != mz.models.Bert else 492,
                truncated_mode="post")

            preprocessor.multiprocessing = 0 if self.dataset.debug_mode else 1
            preprocessor.extra_terms = extra_terms
        return preprocessor, save_path

    def prepare(self, model_class, task=None, config=None, extra_terms=None, experiment: Experiment = None):
        self.model_class = model_class
        self.logger = experiment
        preprocessor, save_path = self._load_basic_preprocessor(extra_terms)
        update_preprocessor = type(self.preprocessor) != type(preprocessor)

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
            print("Transform dataset" )
            self.dataset.set_preprocessor(self.preprocessor)
        if save_path and self.model_class != mz.models.Bert and not self.dataset.debug_mode:
            preprocessor.fit = lambda *args, **argv: None
            self.preprocessor.save(save_path)
            print("Save Preprocessor to %s" % save_path)

    def _reset_model(self, configs=None):
        self.model._params.update(configs)
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
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': configs["weight_decay"]},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = optimizer or configs["optimizer_cls"](optimizer_grouped_parameters,
                                                              lr=configs["lr"], )

        save_dir = save_dir or self.checkpoint_path.joinpath(run_name)
        os.makedirs(save_dir, exist_ok=True)
        model_norm, embedding_norm = calculate_model_norm(self.model)
        self.logger.log_metric(name="model_norm_untrained", value=model_norm)
        self.logger.log_metric(name="embedding_norm_untrained", value=model_norm)

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
        model_norm, embedding_norm = calculate_model_norm(self.model)
        self.logger.log_metric(name="model_norm_trained", value=model_norm)
        self.logger.log_metric(name="embedding_norm_trained", value=embedding_norm)

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
        self.logger.log_asset_data(rounds_dict, file_name="rounds_dict")
        result = {}
        for metric_name, metric_fn in robustness.get_robustness_metrics().items():
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
            "epochs": 20,
            "patience": 3,
            "optimizer_cls": torch.optim.Adam,
            "lr": 3e-4,
            "batch_size": 64,
            "weight_decay": 0,
            "data_aug": 0
        }

    def _get_default_run_name(self, configs):
        name = [self.model_class.__name__]
        return ".".join(name)

    def get_dataloaders(self, configs):
        training_pack = self.dataset.train_pack_processed
        # Setup data
        batch_size = configs.get("batch_size")
        if configs["data_aug"] > 0:
            max_length = 492 if self.model_class == mz.models.Bert else 1024
            callbacks = [InsertQueryToDoc(ratio=configs["data_aug"], max_length=max_length)]
        else:
            callbacks = None

        trainset = self.dataset_builder.build(
            training_pack,
            batch_size=batch_size,
            sort=False,
            callbacks=callbacks
        )
        train_loader = self.dataloader_builder.build(trainset)
        eval_dataset_builder = mz.dataloader.DatasetBuilder(
            batch_size=batch_size * 2,
            shuffle=False,
            sort=False,
            resample=False,
            mode="point"
        )
        dev_dataset = eval_dataset_builder.build(self.dataset.dev_pack_processed)
        dev_loader = self.dataloader_builder.build(dataset=dev_dataset,
                                                   stage="dev")

        return train_loader, dev_loader

    def free_memory(self):
        del self.trainer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
