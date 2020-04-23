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


class ReRankSampler(data.DistributedSampler):
    def __iter__(self):
        rerank_num = 1000
        assert len(self.dataset) % rerank_num == 0
        i = 0
        new_indices = []
        while i < len(self.dataset):
            expected_rank = (i // rerank_num) % self.num_replicas
            if expected_rank == self.rank:
                new_indices.extend(range(i, i + rerank_num))
            i += rerank_num
        return iter(new_indices)


class MsMarcoPL(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.bert_model = hparams.bert_model
        self.distributed = hparams.distributed
        self.batch_size = hparams.batch_size
        self.ms_marco_path = hparams.ms_marco_path

        self.model = transformers.BertForSequenceClassification.from_pretrained(
                self.bert_model, num_labels=1, hidden_dropout_prob=0.1)

        self.loss_fn = mz.losses.RankHingeLoss()
        self.metrics = {
                'ndcg10': mz.metrics.NormalizedDiscountedCumulativeGain(10),
                'ndcg20': mz.metrics.NormalizedDiscountedCumulativeGain(20),
                'mrr': mz.metrics.MeanReciprocalRank(),
                'mrr10': MRRk(10),
                'map': mz.metrics.MeanAveragePrecision()
        }

    def forward(self, x):
        logits = self.model(**x)[0]
        return logits

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

    def validation_epoch_end(self, outputs):
        topics = {}
        for qid, _, y_pred, y_true in itertools.chain(*outputs):
            topics.setdefault(qid, {'pred': [], 'true': []})
            topics[qid]['pred'].append(y_pred.item())
            topics[qid]['true'].append(y_true.item())

        result = {metric_name: [] for metric_name in self.metrics.keys()}
        evaluated = 0
        skipped = 0
        for qid, y in topics.items():
            y_true = y['true']
            y_pred = y['pred']
            if len(y_true) != 1000:
                skipped += 1
                continue
            evaluated += 1
            for metric_name, metric_fn in self.metrics.items():
                result[metric_name].append(metric_fn(y_true=y_true, y_pred=y_pred))
        
        for metric_name, values in result.items():
            result[metric_name] = np.mean(values)

        result['evaluated_topics'] = evaluated
        result['skipped_topics'] = skipped
        return {
                'log': result, 'ndcg10': result['ndcg10'],
                'mrr': result['mrr'], 'map': result['map'],
                'val_loss': result['mrr10']
                }

    def test_step(self, batch, batch_idx):
        return list(self.validation_step(batch, batch_idx))

    def test_epoch_end(self, outputs):
        if len(outputs) == 4:  # qrels are included
            return self.validation_end(outputs)
        topics = {}
        for qid, pid, y_pred in itertools.chain(*outputs):
            topics.setdefault(qid, {'pred': [], 'pid': []})
            topics[qid]['pid'].append(pid.item())
            topics[qid]['pred'].append(y_pred.item())

        version_num = 0
        submission_file = 'result_v{}.txt'
        while os.path.exists(submission_file.format(version_num)):
            version_num += 1
        with open(submission_file, 'w') as f:
            for qid, rankings in topics.items():
                pids = rankings['pid']
                scores = rankings['pred']
                sorted_ranks = sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)
                for rank, (pid, score) in enumerate(sorted_ranks, 1):
                    f.write(f'{qid} Q0 {pid}  {rank} {score} v{version_num}\n')
        # No need to return if writing to file


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
                sampler=None,
                batch_size=self.batch_size,
                collate_fn=MsMarcoPassageTriplesIterableDataset.Collater(self.bert_model),
                num_workers=6
        )

    @pl.data_loader
    def val_dataloader(self):
        dataset = MsMarcoPassageReRankDataset(self.ms_marco_path, mode='dev')
        sampler = ReRankSampler(dataset, shuffle=False) if self.distributed else None
        return data.DataLoader(
                dataset,
                batch_size=self.batch_size * 2,
                sampler=sampler,
                collate_fn=MsMarcoPassageReRankDataset.Collater(self.bert_model),
                num_workers=6
        )

    @pl.data_loader
    def test_dataloader(self):
        dataset = MsMarcoPassageReRankDataset(self.ms_marco_path, mode='test')
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
            max_epochs=1,  gpus=2,
            gradient_clip_val=1, distributed_backend='ddp', use_amp=True,
            early_stop_callback=early_stop_callback,
            checkpoint_callback=checkpoint_callback, logger=logger,
            track_grad_norm=2, num_sanity_val_steps=25, accumulated_grad_batches=4,
            resume_from_checkpoint=args.resume, val_check_interval=400_000, val_percent_check=0.1,
            replace_sampler_ddp=False
        )
        hparams = argparse.Namespace(
            ms_marco_path=ms_marco_path, 
            bert_model='bert-base-uncased',
            batch_size=4,
            distributed=trainer.distributed_backend == 'ddp'
        )
        model = MsMarcoPL(hparams)
        trainer.fit(model)
    else:
        trainer = pl.Trainer(gpus=1, use_amp=True, resume_from_checkpoint=args.resume)
        hparams = argparse.Namespace(
            ms_marco_path=ms_marco_path, 
            bert_model='bert-base-uncased',
            batch_size=8,
            distributed=trainer.distributed_backend == 'ddp'
        )
        model = MsMarcoPL(hparams)
        trainer.test(model)


if __name__ == '__main__':
    main()
