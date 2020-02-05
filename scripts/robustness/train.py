import logging
import os
from pprint import pprint
from comet_ml import Experiment

import matchzoo as mz
from neural_ranking.dataset.asr.asr_collection import AsrCollection
from neural_ranking.matchzoo_helper.dataset import ReRankDataset
from neural_ranking.matchzoo_helper.runner import Runner


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
    parser.add_argument("--saved-preprocessor", type=path, default=None)

    args = parser.parse_args()
    return args

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

    model_classes = [mz.models.MatchLSTM, mz.models.ConvKNRM]
    for model_class in model_classes:
        runner.prepare(model_class, extra_terms=asrc._terms)
        for dropout in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            exp = Experiment(project_name="ASR" if not args.test else "ASR-test",
                             workspace="Robustness",
                             log_env_cpu=False)
            exp.add_tag("%s" % model_class.__name__)
            exp.log_parameter("embedding_name", str(embedding))
            runner.logger = exp
            runner.train(dropout=dropout, dropout_rate=dropout, batch_size=16)
            runner.eval_asrc(asrc)

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    main()
