import logging
import os
from pathlib import Path

import torch

import matchzoo as mz
from neural_ranking.matchzoo_helper.dataset import ReRankDataset
from neural_ranking.matchzoo_helper.utils import dict_mean, ReRankTrainer, get_ranking_task


class Runner(object):
    def __init__(self,
                 dataset: ReRankDataset,
                 embedding=None,
                 log_path: str = "log",
                 checkpoint_path="checkpoint",
                 fp16=False):
        self.dataset = dataset
        self.raw_embedding = embedding

        self.logger = logging.getLogger("Runner")
        self.preprocessor = None
        self.name = None
        self.fp16 = fp16

        self.checkpoint_path = Path(checkpoint_path).absolute()
        self.log_path = Path(log_path).absolute()

    def prepare(self, model_class, task=None, preprocessor=None,
                force_update_preprocessor=True, config=None, extra_terms=None):
        self.model_class = model_class
        self.run_name = None
        preprocessor = preprocessor or self.model_class.get_default_preprocessor(truncated_length_left=20,
                                                                                 truncated_length_right=1024,
                                                                                 truncated_mode="post")
        preprocessor.multiprocessing = 1
        preprocessor.extra_terms = extra_terms
        update_preprocessor = force_update_preprocessor or type(self.preprocessor) != type(preprocessor)
        if not update_preprocessor and self.preprocessor:
            preprocessor = self.preprocessor

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

    def _reset_model(self):
        self.model = self.model_class(params=self.model._params)
        self.model.build()
        return self.model

    def run(self, optimizer=None, optimizer_fn=None, scheduler=None,
            scheduler_fn=None, run_name=None, save_dir=None,
            train=True, devices=None, **kwargs):
        self._reset_model()

        configs = self._get_default_configs()
        configs.update(kwargs)
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
        return preds

    def train_kfold(self, kfold_topic_splits=None, fold_num=None, **kwargs):
        results = []
        kfolds = kfold_topic_splits or self.dataset.get_kfold_splits()
        for i, topic_splits in enumerate(kfolds):
            if fold_num and i >= fold_num:
                break
            print(">>>> Fold - %d of %d" % (i, len(kfolds)))
            self.dataset.init_topic_splits(topic_splits)
            results.append(self.run(**kwargs))
        result = dict_mean(results)
        return result

    def _get_default_configs(self):
        return {
            "epochs": 10,
            "patience": 5,
            "train_ratio": 1.0,
            "optimizer_cls": torch.optim.Adam,
            "lr": 1e-3,
            "batch_size": 64
        }

    def _get_default_run_name(self, configs):
        name = [self.model_class.__name__]
        for k, v in configs.items():
            if isinstance(v, type):
                str_v = v.__name__
            else:
                str_v = str(v)
            name.append("%s=%s" % (str(k), str_v))
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
