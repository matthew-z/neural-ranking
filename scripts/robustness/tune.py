import logging
import os

import comet_ml
import torch

import matchzoo as mz
from neural_ranking.dataset.asr.asr_collection import AsrCollection
from neural_ranking.matchzoo_helper.dataset import ReRankDataset
from neural_ranking.matchzoo_helper.runner import Runner
from scripts.robustness.train import multi_gpu
from itertools import product


def parse_args():
    def path(str):
        return os.path.abspath((os.path.expanduser(str)))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="trec_web.1-200.asrc")
    parser.add_argument("--asrc-path", type=path, default=None)
    parser.add_argument("--log-path", type=path, default="robustness_log")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--gpu-num", type=int, default=1)
    parser.add_argument("--models", type=str, choices=["bert", "others", "match_lstm", "conv_knrm", "all"],
                        default="all")
    parser.add_argument("--saved-preprocessor", type=path, default="preprocessor")
    parser.add_argument("--repeat", type=int, default=5)

    args = parser.parse_args()
    return args


def get_lstm_arguments():
    return {
        "rnn_type": ["lstm", "gru"],
        "hidden_size": [128, 256, 512],
        "lstm_layer": [1,2,3],
        "lr":[1e-3, 1e-4]
    }


def get_ConvKNRM_arguments():
    return {
        "filters": [128, 256, 512],
        "lr":[1e-3, 1e-4],
        "max_ngram": [1,2,3]
    }

def get_bert_arguments():
    return {
        "lr":[1e-3, 1e-4],
    }

def get_params(model_name):
    if model_name=="match_lstm":
        d = get_lstm_arguments()
    elif model_name=="conv_knrm":
        d = get_ConvKNRM_arguments()
    elif model_name=="bert":
        d = get_bert_arguments()
    else:
        raise ValueError()

    for items in product(*d.values()):
        res = {}
        for param_name, value in zip(d.keys(), items):
            res[param_name] = value
        yield res

def main():
    args = parse_args()
    embedding = mz.embedding.GloVe(dim=300, name="840B")
    dataset = ReRankDataset(args.dataset, rerank_hits=100, debug_mode=args.test)
    asrc = AsrCollection(args.asrc_path, dataset.topic_path)
    dataset.init_topic_splits(dev_ratio=0.2, test_ratio=0, seed=2020)
    runner = Runner(embedding=embedding,
                    preprocessor_path=args.saved_preprocessor,
                    log_path=args.log_path,
                    dataset=dataset,
                    fp16=args.fp16)
    if args.models == "bert":
        model_class = mz.models.Bert
    elif args.models == "conv_knrm":
        model_class = mz.models.ConvKNRM
    elif args.models == "match_lstm":
        model_class = mz.models.MatchLSTM
    else:
        raise ValueError("invalid model class")

    for params in get_params(args.models):
        exp = comet_ml.Experiment(project_name="ASR-tune" if not args.test else "ASR-tune-test",
                                  workspace="Robustness",
                                  log_env_cpu=False)
        exp.add_tag("%s" % model_class.__name__)
        exp.add_tag("weight_decay")
        exp.add_tag("tuning")
        exp.log_parameter("embedding_name", str(embedding))
        runner.logger = exp

        runner.prepare(model_class, extra_terms=asrc._terms)
        batch_size = 32 * args.gpu_num if model_class != mz.models.Bert else 3 * args.gpu_num

        runner.train(
            epochs=3 if args.test else 20,
            optimizer_cls=torch.optim.Adam,
            batch_size=batch_size,
            devices=multi_gpu(args.gpu_num if model_class != mz.models.MatchLSTM else 1)
            **params
        )
        runner.free_memory()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
