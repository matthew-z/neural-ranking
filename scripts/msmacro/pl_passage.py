import argparse
import itertools
from pathlib import Path
import os

import matchzoo as mz
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils import data
import transformers

from neural_ranking.data_loader.msmacro.passage import MsMarcoPassageTriplesIterableDataset, MsMarcoPassageReRankDataset


class MRRk:
    def __init__(self, k, threshold=0):
        self.k = k
        self.metric = mz.metrics.MeanReciprocalRank(threshold=threshold)
    def __call__(self, y_true, y_pred):
        return self.metric(y_true=y_true[:self.k], y_pred=y_pred[:self.k])


class MsMarcoPL(pl.LightningModule):
    def __init__(self, ms_marco_path, bert_model='bert-base-uncased',
            distributed=False, batch_size=6):
        super().__init__()
        self.bert_model = bert_model
        self.model = transformers.BertForSequenceClassification.from_pretrained(
                bert_model, num_labels=1, hidden_dropout_prob=0.1)
        self.sigmoid = torch.nn.Sigmoid()

        self.loss_fn = mz.losses.RankHingeLoss()
        self.metrics = {
                'ndcg10': mz.metrics.NormalizedDiscountedCumulativeGain(10),
                'ndcg20': mz.metrics.NormalizedDiscountedCumulativeGain(20),
                'mrr': mz.metrics.MeanReciprocalRank(),
                'mrr10': MRRk(10),
                'map': mz.metrics.MeanAveragePrecision()
        }
        self.ms_marco_path = ms_marco_path
        self.batch_size = batch_size
        self.distributed = distributed

    def forward(self, x):
        logits = self.model(**x)[0]
        return self.sigmoid(logits)

    def training_step(self, batch, batch_index):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_true=y_true, y_pred=y_pred)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_index):
        qid, pid, x, y_true = batch
        y_pred = self.forward(x)
        return zip(qid, pid, y_pred, y_true)

    def validation_end(self, outputs):
        topics = {}
        for qid, _, y_pred, y_true in itertools.chain(*outputs):
            topics.setdefault(qid, {'pred': [], 'true': []})
            topics[qid]['pred'].append(y_pred.item())
            topics[qid]['true'].append(y_true.item())

        result = {}
        evaluated = 0
        skipped = 0
        for qid, y in topics.items():
            if len(y['true']) != 1000:
                skipped += 1
                continue
            evaluated += 1
            for metric_name, metric_fn in self.metrics.items():
                result.setdefault(metric_name, [])
                result[metric_name].append(metric_fn(y_true=y_true, y_pred=y_pred))
        
        for metric_name, values in result.items():
            result[metric_name] = np.mean(values)

        if not result:  # 
            result = {
                    metric_name: 0 for metric_name in self.metrics.keys()
            }
        
        result['evaluated_topics'] = evaluated
        result['skipped_topics'] = skipped
        return {
                'log': result, 'ndcg10': result['ndcg10'],
                'mrr': result['mrr'], 'map': result['map'],
                'val_loss': result['mrr10']
                }

    def configure_optimizers(self):
        lr = 3e-6
        num_training_steps = 200_000
        num_warmup_steps = 10_000
        optimizer = transformers.AdamW(self.parameters(), lr=lr)
        scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
        )
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=1,
                factor=0.5, threshold=0.05,
                threshold_mode='abs'
        )
        return [optimizer], [scheduler, reduce_lr]

    @pl.data_loader
    def train_dataloader(self):
        dataset = MsMarcoPassageTriplesIterableDataset(self.ms_marco_path)
        return data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=MsMarcoPassageTriplesIterableDataset.Collater(self.bert_model),
                num_workers=6
        )

    @pl.data_loader
    def val_dataloader(self):
        dataset = MsMarcoPassageReRankDataset(self.ms_marco_path)
        return data.DataLoader(
                dataset,
                batch_size=self.batch_size * 2,
                collate_fn=MsMarcoPassageReRankDataset.Collater(self.bert_model),
                num_workers=6
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    monitor = 'mrr'
    project_folder = os.path.join(os.getcwd(), 'msmarco_log')
    os.makedirs(project_folder, exist_ok=True)

    logger = pl.loggers.TensorBoardLogger(
            save_dir=project_folder,
            name='bert-base-uncased'
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
            monitor=monitor,
            patience=5,
            verbose=True,
            mode='max'
    )
    ckpt_path = os.path.join(logger.save_dir, logger.name,
            'version_{}'.format(logger.version), 'checkpoint')
    os.makedirs(ckpt_path, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_top_k=1,
            verbose=True,
            monitor=monitor,
            mode='max',
            prefix='bert_base_uncase'
    )

    ms_marco_path = Path('~/neural-ranking/datasets/msmarco').expanduser()

    if not args.test:
        trainer = pl.Trainer(
                min_epochs=3, max_epochs=10, gpus=2, train_percent_check=0.03,
                gradient_clip_val=1, distributed_backend='ddp', use_amp=True,
                early_stop_callback=early_stop_callback,
                checkpoint_callback=checkpoint_callback, logger=logger,
                track_grad_norm=2, num_sanity_val_steps=25, accumulated_grad_batches=4,
                resume_from_checkpoint=args.resume, val_check_interval=1.0)
        model = MsMarcoPL(ms_marco_path, batch_size=4, 
                distributed=trainer.distributed_backend == 'ddp')
        trainer.fit(model)
    else:
        trainer = pl.Trainer(
                gpus=1, use_amp=True, resume_from_checkpoint=args.resume, 
                test_percent_check=0.1)
        model = MsMarcoPL(ms_marco_path, batch_size=8,
                distributed=trainer.distributed_backend == 'ddp')
        trainer.test(model)


if __name__ == '__main__':
    main()
