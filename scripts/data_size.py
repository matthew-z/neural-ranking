import logging
import os
import matchzoo as mz

from neural_ranking.matchzoo_helper.runner_pytorch import Runner, ReRankDataset


def path(str):
    return os.path.abspath((os.path.expanduser(str)))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="robust04")
    parser.add_argument("--log-path", type=path, default="data_size_log")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    embedding = mz.embedding.GloVe(dim=50, name="6B")

    dataset = ReRankDataset(args.dataset)
    runner = Runner(embedding=embedding,
                    log_path=args.log_path,
                    dataset=dataset)
    for model_class in [mz.models.ConvKNRM, mz.models.KNRM]:
        for train_ratio in [1, 0.75, 0.5, 0.25]:
            print("Prepare for Model %s" % model_class)
            runner.prepare(model_class)
            print(">>>>>>> Next Phase: train ratio = %.2f" % train_ratio)
            runner.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
