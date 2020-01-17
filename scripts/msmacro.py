import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, DistributedSampler

import matchzoo as mz
from neural_ranking.dataset.msmacro import MsMacroTriplesDataset, MsMacroReRankDataset


class MsMacro(pl.LightningModule):

    def __init__(self, ms_macro_path, bert_model="bert-base-uncased"):
        super().__init__()
        self.bert_model = bert_model
        self.model = transformers.BertForSequenceClassification.from_pretrained(bert_model, num_labels=1)

        self.loss_fn = mz.losses.RankHingeLoss()
        self.metrics = {
            "ndcg@10": mz.metrics.NormalizedDiscountedCumulativeGain(10),
            "mrr": mz.metrics.MeanReciprocalRank(),
            "map": mz.metrics.MeanAveragePrecision()
        }
        self.main_metric_name = "ndcg@10"
        self.ms_macro_path = ms_macro_path
        self.batch_size = 6

    def forward(self, x):
        logits = self.model(**x)[0]
        score = torch.nn.functional.logsigmoid(logits)
        return score

    def training_step(self, batch, batch_idx):
        topic_ids, doc_ids, x, y_true = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred=y_pred, y_true=y_true)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        topic_ids, doc_ids, x, y_true = batch
        y_pred = self.forward(x)
        return topic_ids, doc_ids, y_pred, y_true

    def validation_end(self, outputs):
        topics = {}
        for topic_id, doc_id, y_pred, y_true in outputs:
            topics.setdefault(topic_id, {"pred": [], "true": []})
            topics[topic_id]["pred"].append(y_pred)
            topics[topic_id]["true"].append(y_true)

        result = {}
        for topic_id, y in topics.items():
            for metric_name, metric_fn in self.metrics.items():
                result.setdefault(metric_name, [])
                result[metric_name].append(metric_fn(y_true=y["true"], y_pred=y["pred"]))

        for metric_name, values in result.items():
            result[metric_name] = np.mean(values)
        return {'log': result}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return transformers.AdamW(self.parameters())

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        triples_path = os.path.join(self.ms_macro_path, "triples.tsv")
        dataset = MsMacroTriplesDataset(triples_path)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                          num_workers=8, collate_fn=MsMacroTriplesDataset.Collater(self.bert_model))

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        dataset = MsMacroReRankDataset(self.ms_macro_path)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                          num_workers=8, collate_fn=MsMacroReRankDataset.Collater(self.bert_model))


def main():
    ms_macro_path = Path("~/ir/collection/ms-macro/document").expanduser()
    model = MsMacro(ms_macro_path)
    # most basic trainer, uses good defaults
    trainer = Trainer(max_nb_epochs=1, val_check_interval=500, gpus=-1, distributed_backend="ddp", use_amp=True,
                      amp_level="O2")
    trainer.fit(model)


if __name__ == "__main__":
    main()
