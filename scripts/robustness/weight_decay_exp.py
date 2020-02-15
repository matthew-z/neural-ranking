import logging
import os
import comet_ml
import torch
from test_tube import HyperOptArgumentParser

import matchzoo as mz
from neural_ranking.dataset.asr.asr_collection import AsrCollection
from neural_ranking.matchzoo_helper.dataset import ReRankDataset
from neural_ranking.matchzoo_helper.runner import Runner


def parse_args():
    def path(str):
        return os.path.abspath((os.path.expanduser(str)))

    parser = HyperOptArgumentParser()
    parser.add_argument("--dataset", type=str, default="trec_web.1-200.asrc")
    parser.add_argument("--asrc-path", type=path, default=None)
    parser.add_argument("--log-path", type=path, default="robustness_log")
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--gpu-num", type=int, default=1)
    parser.add_argument("--model", type=str, choices=["bert", "mp", "conv_knrm", "all"],
                        default="all")
    parser.add_argument("--exp", type=str, default="weight_decay", choices=["dropout", "weight_decay"])
    parser.add_argument("--saved-preprocessor", type=path, default="preprocessor")
    parser.opt_list("--weight-decay", type=float, tunable=True,
                        options=[0.0001, 0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20])
    args = parser.parse_args()
    return args


def main(args, gpu_ids):
    embedding = mz.embedding.GloVe(dim=300, name="840B")
    dataset = ReRankDataset(args.dataset, rerank_hits=100, debug_mode=args.test)
    asrc = AsrCollection(args.asrc_path, dataset.topic_path)
    dataset.init_topic_splits(dev_ratio=0.2, test_ratio=0, seed=2020)
    runner = Runner(embedding=embedding,
                    preprocessor_path=args.saved_preprocessor,
                    log_path=args.log_path,
                    dataset=dataset,
                    fp16=args.fp16)

    model_class = parse_model_class(args.model)

    if args.exp == "dropout":
        exp = dropout_exp
    elif args.exp == "weight_decay":
        exp = weight_decay_exp
    else:
        raise ValueError("Incorrect Exp Value: %s" % args.exp)

    exp(args, asrc, embedding, model_class, runner)


def parse_model_class(model_name):
    if model_name == "bert":
        model_class = mz.models.Bert
    elif model_name == "others":
        model_class = mz.models.MatchPyramid, mz.models.ConvKNRM, mz.models.MatchLSTM
    elif model_name == "conv_knrm":
        model_class = mz.models.ConvKNRM
    elif model_name == "match_lstm":
        model_class = mz.models.MatchLSTM
    elif model_name == "mp":
        model_class = mz.models.MatchPyramid
    else:
        raise ValueError("invalid model class %s" % model_name)
    return model_class


def multi_gpu(gpu_num=1):
    if gpu_num == 0:
        return "cpu"
    else:
        return [torch.device('cuda:%d' % i) for i in range(gpu_num)]


def weight_decay_exp(args, asrc, embedding, model_class, runner: Runner):
    weight_decay = args.weight_decay
    exp = comet_ml.Experiment(project_name="ASR" if not args.test else "ASR-test",
                              workspace="Robustness",
                              log_env_cpu=False)
    exp.add_tag("%s" % model_class.__name__)
    exp.add_tag("weight_decay")
    exp.log_parameter("embedding_name", str(embedding))
    runner.prepare(model_class, extra_terms=asrc._terms)
    runner.logger = exp
    batch_size = 32 * args.gpu_num if model_class != mz.models.Bert else 2 * args.gpu_num
    runner.train(
        epochs=3 if args.test else 20,
        weight_decay=weight_decay,
        optimizer_cls=torch.optim.AdamW,
        batch_size=batch_size,
        lr=3e-4 if model_class != mz.models.Bert else 3e-5,
        devices=multi_gpu(args.gpu_num if model_class != mz.models.MatchLSTM else 1)
    )
    runner.eval_asrc(asrc)
    runner.free_memory()


def dropout_exp(args, asrc, embedding, model_class, runner: Runner):
    exp = comet_ml.Experiment(project_name="ASR" if not args.test else "ASR-test",
                              workspace="Robustness",
                              log_env_cpu=False)
    exp.add_tag("%s" % model_class.__name__)
    exp.add_tag("dropout")
    exp.log_parameter("embedding_name", str(embedding))
    runner.prepare(model_class, extra_terms=asrc._terms)
    runner.logger = exp
    batch_size = 32 * args.gpu_num if model_class != mz.models.Bert else 3 * args.gpu_num
    runner.train(
        epochs=3 if args.test else 20,
        dropout=args.dropout,
        dropout_rate=args.dropout,
        batch_size=batch_size,
        lr=3e-4 if model_class != mz.models.Bert else 3e-5,
        devices=multi_gpu(args.gpu_num if model_class != mz.models.MatchLSTM else 1)
    )
    runner.eval_asrc(asrc)
    runner.free_memory()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    hparams = parse_args()
    hparams.optimize_parallel_gpu(main, gpu_ids=["0"])
