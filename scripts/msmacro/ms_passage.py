from itertools import chain
from pathlib import Path
import json
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import matchzoo as mz
from neural_ranking.data_loader.msmacro.passage import MsMarcoPassageTriplesIterableDataset, MsMarcoPassageReRankDataset


class MRRk:
    def __init__(self, k, threshold=0):
        self.k = k
        self.metric = mz.metrics.MeanReciprocalRank(threshold=threshold)

    def __call__(self, y_true, y_pred):
        return self.metric(y_true=y_true[:self.k], y_pred=y_pred[:self.k])


class MsMacro(torch.nn.Module):
    def __init__(self, ms_macro_path, bert_model="bert-base-uncased",
                 distributed=False, batch_size=6, concat_query=False):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(
                bert_model, num_labels=1, hidden_dropout_prob=0.1
            )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        logits = self.model(**x)[0]
        return self.sigmoid(logits)

def training_step(model, loss_fn, batch):
    x, y_true = batch
    y_pred = model(x)
    loss = loss_fn(y_pred=y_pred, y_true=y_true)
    return loss

def test_step(model, batch):
    qid, pid, x, y_true = batch
    y_pred = model(x)
    return zip(qid, pid, y_pred, y_true)

def configure_optimizers(model):
    lr = 3e-6
    num_training_steps = 200000
    num_warmup_steps = 10000
    optimizer = transformers.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    return optimizer, scheduler

def train_dataloader(ms_macro_path, batch_size, distributed, bert_model):
    dataset = MsMarcoPassageTriplesIterableDataset(ms_macro_path)
    sampler = DistributedSampler(dataset) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
            collate_fn=MsMarcoPassageTriplesIterableDataset.Collater(bert_model),
            num_workers=6, shuffle=False)

def test_dataloader(ms_macro_path, batch_size, bert_model):
    dataset = MsMarcoPassageReRankDataset(ms_macro_path)
    return DataLoader(dataset, batch_size=batch_size * 2,
                      collate_fn=MsMarcoPassageReRankDataset.Collater(bert_model))


def main():
    ms_macro_path = Path("~/neural_ranking/built_data/msmarco").expanduser()
    distributed = False
    batch_size = 4
    bert_model = "bert-base-uncased"
    loss_fn = mz.losses.RankHingeLoss()


    net = MsMacro(ms_macro_path, batch_size=4, distributed=distributed)
    optimizer, scheduler = configure_optimizers(net)

    print('Starting to train')
    trainset = train_dataloader(ms_macro_path, batch_size, distributed, bert_model)
    for epoch in range(0):
        net.train()
        for batch_idx, x in enumerate(trainset):
            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = training_step(net, loss_fn, x)
            loss.backward()
            optimizer.step()
            scheduler.step()
          
            print("Epoch {}: [Iter {}] loss: {}".format(epoch, (batch_idx + 1) * len(x[1] / 2), loss.item()))
  
            if batch_idx == 2:
                break

    print('Starting to evaluate')
    testset = test_dataloader(ms_macro_path, batch_size * 2, bert_model)

    net.eval()
    topics = {}
    count = 0
    with torch.no_grad():
        for batch_idx, x in enumerate(testset): 
            output = test_step(net, x)
            for qid, _, pred, actual in output:
                count += 1
                topics.setdefault(qid, {"pred": [], "true": []})
                topics[qid]["pred"].append(pred.item())
                topics[qid]["true"].append(actual.item())

    result = {'mrr': [], 'mrr2': [], 'skipped_topics': [], 'total_rows': count}
    mrr = MRRk(10)
    for qid, y in topics.items():
        y_true = y["true"]
        y_pred = y["pred"]

        result['mrr2'].append(mrr(y_true=y_true, y_pred=y_pred))
        if len(y_true) != 1000:
            result['skipped_topics'].append((qid, len(y_true)))
            continue
        result['mrr'].append(mrr(y_true=y_true, y_pred=y_pred))

    result['mrr'] = np.mean(result['mrr'])
    result['mrr2'] = np.mean(result['mrr2'])
    
    with open('msmarco_eval.json', 'w') as f:
        json.dump(result, f)

    print('end step')
    
if __name__ == '__main__':
    main()
