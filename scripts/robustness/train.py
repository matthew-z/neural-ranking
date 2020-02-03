import logging
import os
import pickle

from transformers import AdamW

import matchzoo as mz
from neural_ranking.matchzoo_helper.dataset import ReRankDataset
from neural_ranking.matchzoo_helper.runner import Runner


def parse_args():
    def path(str):
        return os.path.abspath((os.path.expanduser(str)))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="robust04")
    parser.add_argument("--log-path", type=path, default="data_size_log")
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    embedding = mz.embedding.GloVe(dim=50, name="6B")
    dataset = ReRankDataset(args.dataset, rerank_hits=100, test=args.test)
    dataset.init_topic_splits(dev_ratio=0.2, test_ratio=0, seed=2020)
    runner = Runner(embedding=embedding,
                    log_path=args.log_path,
                    dataset=dataset,
                    fp16=args.fp16)

    model_classes = [mz.models.MatchLSTM, mz.models.KNRM]

    for model_class in model_classes:
        runner.prepare(model_class)



if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
