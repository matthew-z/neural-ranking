import json

import neural_ranking
import matchzoo as mz
import copy
import torch

from neural_ranking.runners.dataset import ReRankDataset
from neural_ranking.runners.utils import  ReRankTrainer


ranking_task = mz.tasks.Ranking(mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=10),
    mz.metrics.Precision(k=30),
]

print('---starting to process robust04---')
dataset = ReRankDataset("robust04", rerank_hits=1000) # debug mode will only load 100 docs from the dataset
dataset.init_topic_splits(dev_ratio=0.2, test_ratio=0, seed=2020) # split data into train and dev randomly

model,preprocessor,dataset_builder, dataloader_builder = mz.auto.prepare(
            task=ranking_task,
            model_class=mz.models.Bert,
            data_pack=dataset.pack,
            embedding=None, # Bert does not need embedding
            preprocessor=mz.models.Bert.get_default_preprocessor())

dataset.apply_preprocessor(preprocessor)

def get_dataloaders(dataset, dataset_builder, dataloader_builder, batch_size=2):
    training_pack = dataset.train_pack_processed
    # Setup data
    trainset = dataset_builder.build(
        training_pack,
        batch_size=batch_size,
        sort=False,
    )
    train_loader = dataloader_builder.build(trainset)

    eval_dataset_kwargs = copy.copy(dataset_builder._kwargs)
    eval_dataset_kwargs["batch_size"] = batch_size * 2
    eval_dataset_kwargs["shuffle"] = False
    eval_dataset_kwargs["sort"] = False
    eval_dataset_kwargs["resample"] = False
    eval_dataset_kwargs["mode"] = "point"

    eval_dataset_builder = mz.dataloader.DatasetBuilder(
        **eval_dataset_kwargs,
    )
    
    dev_dataset = eval_dataset_builder.build(dataset.dev_pack_processed)
    dev_loader = dataloader_builder.build(dataset=dev_dataset, stage="dev")
    return train_loader, dev_loader

train_loader, dev_loader = get_dataloaders(dataset, dataset_builder, dataloader_builder)
print('---finished processing robust04---')

print('---starting to process wwww2---')
dataset_test = ReRankDataset("www2", rerank_hits=1000)
dataset_test.init_topic_splits(dev_ratio=0, test_ratio=1, seed=2020)
dataset_test.apply_preprocessor(preprocessor)

testset = dataset_builder.build(
    dataset_test.test_pack_processed,
    batch_size=2,
    sort=False,
)
test_loader = dataloader_builder.build(testset, stage="dev")
print('---finished processing www2---')

optimizer = torch.optim.AdamW(model.parameters())
trainer = ReRankTrainer(
            model=model,
            optimizer=optimizer,
            trainloader=train_loader,
            validloader=dev_loader,
            epochs=5,
            patience=2,
            device="cuda",
            save_dir="checkpoint",
            fp16=False,
            clip_norm=5,
            batch_accumulation=2)

print('---starting training---')
trainer.run()
print('---finished training---')

print('---starting evaluation---')
res = trainer.evaluate(test_loader)
print('---finished evaluation---')
print(res)

with open('www2.json', 'w') as f:
    json.dump({str(k): str(v) for k, v in res.items()}, f)
