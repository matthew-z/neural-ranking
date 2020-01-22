import argparse
import os
from itertools import chain
from pathlib import Path
from pprint import pprint
import json
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup

import matchzoo as mz
import pytorch_lightning as pl
from neural_ranking.dataset.msmacro import MsMacroDocTriplesDataset, MsMacroDocReRankDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger


class MRRk():
    def __init__(self, k, threshold=0):
        self.k = k
        self.metric = mz.metrics.MeanReciprocalRank(threshold=threshold)

    def __call__(self, y_true, y_pred):
        return self.metric(y_true=y_true[:self.k], y_pred=y_pred[:self.k])


class ReRankSampler(DistributedSampler):
    def __iter__(self):
        rerank_num = 100
        # deterministically shuffle based on epoch
        assert len(self.dataset) % rerank_num == 0
        i = 0
        new_indices = []
        while i < len(self.dataset):
            expected_rank = (i // rerank_num) % self.num_replicas
            if expected_rank == self.rank:
                new_indices.extend(range(i, i + rerank_num))
            i += rerank_num
        return iter(new_indices)


class MsMacro(pl.LightningModule):

    def __init__(self, ms_macro_path, bert_model="bert-base-uncased",
                 distributed=False, batch_size=6, concat_query=False):
        super().__init__()
        self.bert_model = bert_model
        self.model = transformers.BertForSequenceClassification.from_pretrained(bert_model,
                                                                                num_labels=1,
                                                                                hidden_dropout_prob=0.1)

        self.loss_fn = mz.losses.RankHingeLoss()
        self.metrics = {
            "ndcg10": mz.metrics.NormalizedDiscountedCumulativeGain(10),
            "ndcg20": mz.metrics.NormalizedDiscountedCumulativeGain(20),
            "mrr": mz.metrics.MeanReciprocalRank(),
            "mrr10": MRRk(10),
            "map": mz.metrics.MeanAveragePrecision()
        }
        self.ms_macro_path = ms_macro_path
        self.batch_size = batch_size
        self.distributed = distributed
        self.distinct_topics = set()
        self.concat_query = concat_query
        if concat_query:
            print("Will concat Query to Doc")

    def forward(self, x):
        logits = self.model(**x)[0]
        return logits

    def training_step(self, batch, batch_idx):
        topic_ids, doc_ids, x, y_true = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred=y_pred, y_true=y_true)
        self.distinct_topics.update(set(topic_ids))
        tensorboard_logs = {'train_loss': loss, "distinct_topics": len(self.distinct_topics)}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        topic_ids, doc_ids, x, y_true = batch
        y_pred = self.forward(x)
        return zip(topic_ids, doc_ids, y_pred, y_true)

    def validation_end(self, outputs):
        topics = {}
        for topic_id, doc_id, y_pred, y_true in chain(*outputs):
            topics.setdefault(topic_id, {"pred": [], "true": []})
            topics[topic_id]["pred"].append(y_pred.item())
            topics[topic_id]["true"].append(y_true.item())

        result = {}
        evaluated = 0
        skipped = 0
        for topic_id, y in topics.items():
            y_true = y["true"]
            y_pred = y["pred"]
            if len(y_true) != 100:
                skipped += 1
                continue
            evaluated += 1
            for metric_name, metric_fn in self.metrics.items():
                result.setdefault(metric_name, [])
                result[metric_name].append(metric_fn(y_true=y_true, y_pred=y_pred))

        for metric_name, values in result.items():
            result[metric_name] = np.mean(values)

        result["evaluated_topics"] = evaluated
        result["skipped_topics"] = skipped
        return {'log': result, 'ndcg10': result["ndcg10"],
                'mrr': result["mrr"], 'map': result["map"],
                "val_loss": result["mrr10"]}

    def test_step(self, batch, batch_idx):
        return list(self.validation_step(batch, batch_idx))

    def test_end(self, outputs):
        topics = {}
        for topic_id, doc_id, y_pred, y_true in chain(*outputs):
            topics.setdefault(topic_id, {"pred": [], "true": []})
            topics[topic_id]["pred"].append(y_pred.item())
            topics[topic_id]["true"].append(y_true.item())

        result = {}
        evaluated = 0
        skipped = 0
        topic_ids = []
        for topic_id, y in topics.items():
            y_true = y["true"]
            y_pred = y["pred"]
            if len(y_true) != 100:
                skipped += 1
                continue
            evaluated += 1
            result.setdefault("topic_ids", [])
            topic_ids.append(topic_id)
            for metric_name, metric_fn in self.metrics.items():
                result.setdefault(metric_name, [])
                result[metric_name].append(metric_fn(y_true=y_true, y_pred=y_pred))

        filename = "/tmp/metric_concat" if self.concat_query else "/tmp/metric"
        with open(filename, "w") as f:
            json.dump([result,topic_ids], f)

        for metric_name, values in result.items():
            result[metric_name] = np.mean(values)

        result["evaluated_topics"] = evaluated
        result["skipped_topics"] = skipped

        pprint(result)
        return {'log': result, 'ndcg10': result["ndcg10"],
                'mrr': result["mrr"], 'map': result["map"],
                "val_loss": result["mrr10"]}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        lr = 1e-4
        num_training_steps = 200000
        num_warmup_steps = 10000
        optimizer = transformers.AdamW(self.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1,
                                                               factor=0.1, threshold=0.05,
                                                               threshold_mode="abs")
        return [optimizer], [scheduler, reduce_lr]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        dataset = MsMacroDocTriplesDataset(self.ms_macro_path)
        sampler = DistributedSampler(dataset) if self.distributed else None
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                          collate_fn=MsMacroDocTriplesDataset.Collater(self.bert_model),
                          num_workers=6,
                          shuffle=True if sampler is None else False)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        dataset = MsMacroDocReRankDataset(self.ms_macro_path)
        sampler = ReRankSampler(dataset, shuffle=False) if self.distributed else None
        return DataLoader(dataset, batch_size=self.batch_size * 2,
                          sampler=sampler,
                          collate_fn=MsMacroDocReRankDataset.Collater(self.bert_model),
                          num_workers=6)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        dataset = MsMacroDocReRankDataset(self.ms_macro_path, concat_query=self.concat_query)
        return DataLoader(dataset, batch_size=self.batch_size * 2,
                          collate_fn=MsMacroDocReRankDataset.Collater(self.bert_model),
                          num_workers=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--concat-query", action='store_true')

    args = parser.parse_args()

    monitor = "mrr"
    os.makedirs(os.path.join(os.getcwd(), 'msmacro_log'), exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name='msmacro_log',
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        patience=3,
        verbose=True,
        mode='max'
    )
    ckpt_path = os.path.join(logger.save_dir, logger.name, "version_%d" % logger.version, "checkpoint")
    os.makedirs(ckpt_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=1,
        verbose=True,
        monitor=monitor,
        mode='max',
        prefix='bert_base_uncase'
    )

    ms_macro_path = Path("~/ir/collection/ms-macro/document").expanduser()

    if not args.test:
        trainer = Trainer(min_epochs=20, max_epochs=50, gpus=2, train_percent_check=0.03,
                          gradient_clip_val=1, distributed_backend="ddp", use_amp=True,
                          early_stop_callback=early_stop_callback, checkpoint_callback=checkpoint_callback,
                          logger=logger,
                          track_grad_norm=2, num_sanity_val_steps=25, accumulate_grad_batches=4,
                          resume_from_checkpoint=args.resume)

        model = MsMacro(ms_macro_path, batch_size=4, distributed=trainer.distributed_backend == "ddp")
        trainer.fit(model)

    else:
        trainer = Trainer(gpus=1,use_amp=True, resume_from_checkpoint=args.resume, test_percent_check=0.1)
        model = MsMacro(ms_macro_path, batch_size=8,
                        distributed=trainer.distributed_backend == "ddp",
                        concat_query =args.concat_query)
        trainer.test(model)

if __name__ == "__main__":
    main()