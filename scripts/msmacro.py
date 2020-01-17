from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup

import matchzoo as mz
import pytorch_lightning as pl
from neural_ranking.dataset.msmacro import MsMacroTriplesDataset, MsMacroReRankDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


class MsMacro(pl.LightningModule):

    def __init__(self, ms_macro_path, bert_model="bert-base-uncased", distributed=False, batch_size=6):
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
        self.batch_size = batch_size
        self.distributed = distributed
        self._s = None

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
            y_true = torch.cat(y["true"])
            y_pred = torch.cat(y["pred"])
            for metric_name, metric_fn in self.metrics.items():
                result.setdefault(metric_name, [])
                result[metric_name].append(metric_fn(y_true=y_true, y_pred=y_pred))

        for metric_name, values in result.items():
            result[metric_name] = np.mean(values)

        val_ndcg10 = result["ndcg@10"]
        return {'log': result, 'ndcg@10': val_ndcg10}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        lr = 1e-3
        num_training_steps = len(self.train_dataloader())
        num_warmup_steps = 1000
        optimizer = transformers.AdamW(self.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        return [optimizer], [scheduler]
    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        dataset = MsMacroTriplesDataset(self.ms_macro_path)
        sampler = DistributedSampler(dataset) if self.distributed else None
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                          collate_fn=MsMacroTriplesDataset.Collater(self.bert_model),
                          shuffle=True if not sampler else None,
                          num_workers=4
                          )

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        dataset = MsMacroReRankDataset(self.ms_macro_path)
        sampler = DistributedSampler(dataset, shuffle=False) if self.distributed else None
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                          collate_fn=MsMacroReRankDataset.Collater(self.bert_model),
                          num_workers=4)


def main():
    early_stop_callback = EarlyStopping(
        monitor='ndcg@10',
        patience=3,
        verbose=True,
        mode='min'
    )

    ms_macro_path = Path("~/ir/collection/ms-macro/document").expanduser()
    trainer = Trainer(min_nb_epochs=5, max_nb_epochs=10, gpus=[1, 2],
                      gradient_clip_val=1, overfit_pct=0.001, distributed_backend="ddp", use_amp=True, amp_level="O3",
                      early_stop_callback=early_stop_callback)

    model = MsMacro(ms_macro_path, batch_size=4, distributed=trainer.distributed_backend == "ddp")

    # trainer = Trainer(max_nb_epochs=10,  gpus=-1, distributed_backend="ddp",
    #                   gradient_clip_val=1, overfit_pct=0.0001,
    #                   amp_level="O2", use_amp=True, early_stop_callback=early_stop_callback)
    trainer.fit(model)


if __name__ == "__main__":
    main()
