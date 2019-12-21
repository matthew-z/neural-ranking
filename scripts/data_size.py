import logging
import os
import pickle

from transformers import AdamW, get_linear_schedule_with_warmup

import matchzoo as mz
from neural_ranking.matchzoo_helper.runner_pytorch import Runner, ReRankDataset


def path(str):
    return os.path.abspath((os.path.expanduser(str)))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="robust04")
    parser.add_argument("--log-path", type=path, default="data_size_log")
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    return args

def bert_optimizer_fn(model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 5e-5},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    return AdamW(optimizer_grouped_parameters, lr=2e-5, betas=(0.9, 0.98), eps=1e-8)


def main():
    args = parse_args()
    # embedding = mz.embedding.GloVe(dim=50, name="6B")
    # embedding = mz.embedding.GloVe(dim=300, name="840B")
    dataset = ReRankDataset(args.dataset, rerank_hits=100, test=args.test)
    runner = Runner(embedding=None,
                    log_path=args.log_path,
                    dataset=dataset,
                    fp16=args.fp16)
    scores = {}

    for model_class in [mz.models.Bert, mz.models.MatchLSTM, mz.models.KNRM]:
        for i, train_ratio in enumerate([1, 0.75, 0.5, 0.25]):
            print("Prepare for Model %s" % model_class)
            runner.prepare(model_class, update_preprocessor=False)
            print(">>>>>>> Next Phase: train ratio = %.2f" % train_ratio)
            if model_class == mz.models.Bert:
                batch_size = 8
                scheduler_fn = lambda optimizer: get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=6,
                    num_training_steps=runner.dataset.train_pack_positive_num / batch_size)
                optimizer_fn = bert_optimizer_fn
            else:
                optimizer_fn = None
                scheduler_fn = None

            score = runner.run_kfold(train_ratio=train_ratio, batch_size=3, epochs=50, patience=3,
                                     optimizer_fn=optimizer_fn, scheduler_fn=scheduler_fn)

            scores.setdefault(model_class, {})
            scores[model_class][train_ratio] = score

    pickle.dump(scores, open("logs/data_size_result.json", "wb"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
