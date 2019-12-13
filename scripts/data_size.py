import os
import matchzoo as mz

from neural_ranking.embedding.glove import load_glove_embedding
from neural_ranking.matchzoo_helper.runner_pytorch import Runner
from neural_ranking.utils import input_check


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
    # embedding = load_glove_embedding(50, "6B")
    runner = Runner(embedding=embedding,
                    log_path=args.log_path,
                    dataset=args.dataset)
    for model_class in [mz.models.ConvKNRM, mz.models.KNRM]:
        for train_ratio in [1, 0.75, 0.5, 0.25]:
            print("Prepare for Model %s" % model_class)
            runner.prepare_model(model_class, train_ratio=train_ratio)
            print(">>>>>>> Next Phase: train ratio = %.2f" % train_ratio)
            runner.run()


if __name__ == "__main__":
    main()
