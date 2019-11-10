import os
import pickle
from pathlib import Path

import keras
import matchzoo as mz
import tqdm
from matchzoo.losses import RankHingeLoss

from neural_ranking.embedding.glove import load_glove_embedding
from neural_ranking.evaluation import EvaluateRankingMetrics, \
    reranking_evalaute
from neural_ranking.utils import load_rerank_data, \
    get_rerank_packs
from neural_ranking.utils.common import split_datapack

DATA_FOLDER = Path("~/ir/neural_ranking/built_data/").expanduser()


class Runner(object):
    def __init__(self, dataset="robust04", embedding=None, log_path="log"):
        self.dataset = dataset
        self.prepare_data(DATA_FOLDER.joinpath(dataset))

        if embedding is None:
            self.raw_embedding = load_glove_embedding(dimension=50, size="6B")
        else:
            self.raw_embedding = embedding

        self.log_path = Path(log_path).absolute()

    def prepare_data(self, datapath):
        pack = mz.load_data_pack(datapath).shuffle()
        rerank_packs = load_rerank_data(datapath)
        self.train_pack, valid_pack, test_pack = split_datapack(pack)
        self.valid_rerank_packs = get_rerank_packs(rerank_packs, valid_pack)
        self.test_rerank_packs = get_rerank_packs(rerank_packs, test_pack)

    def prepare_model(self, model_class, train_ratio=1.0, task=None):
        self.train_ratio = train_ratio
        self.task = task or get_ranking_task()
        self.model_class = model_class
        self.model, self.preprocessor, self.data_gen_builder, self.embedding_matrix = mz.auto.prepare(
            task=self.task,
            model_class=model_class,
            data_pack=self.train_pack,
            embedding=self.raw_embedding
        )
        train_size = int(len(self.train_pack) * train_ratio)

        self.train_pack_processed = self.preprocessor.transform(
            self.train_pack[:train_size])

        self.valid_packs_processed = [
            self.preprocessor.transform(pack, verbose=0)
            for pack in tqdm.tqdm(self.valid_rerank_packs,
                                  desc="processing validation set ")]

        self.test_packs_processed = [
            self.preprocessor.transform(pack, verbose=0)
            for pack in
            tqdm.tqdm(self.test_rerank_packs, desc="processing test set ")]

    def run(self,
            epochs=10,
            train_gen_mode='pair',
            train_gen_num_dup=5, train_gen_num_neg=1,
            batch_size=64):

        train_generator = self.data_gen_builder.build(
            self.train_pack_processed)
        valid_evaluate = EvaluateRankingMetrics(self.model,
                                                test_packs=self.valid_packs_processed,
                                                batch_size=128)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='normalized_discounted_cumulative_gain@5(0.0)',
            min_delta=0, patience=5, verbose=1, mode='max',
            baseline=None, restore_best_weights=True)

        history = self.model.fit_generator(train_generator, epochs=epochs,
                                           workers=30,
                                           use_multiprocessing=True,
                                           callbacks=[valid_evaluate,
                                                      early_stopping])

        # evaluate
        evaluate_test = reranking_evalaute(
            [pack.unpack() for pack in self.test_packs_processed], self.model)
        model_name = self.model.__class__.__name__.split(".")[-1]
        history_name = "%s_%s_%.2f.result.pkl" % (
            model_name, self.dataset, self.train_ratio)

        # write result and history to a file
        result = {"test": evaluate_test, "history": history.history}
        pickle.dump(result, open(self.log_path.joinpath(history_name), "wb"))
        return result


def path(str):
    return os.path.abspath((os.path.expanduser(str)))


def get_ranking_task(loss=None):
    ranking_task = mz.tasks.Ranking(loss=loss or RankHingeLoss())
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        "map",
    ]
    return ranking_task


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="robust04")
    parser.add_argument("--log-path", type=path, default="data_size_log")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # embedding = load_glove_embedding(300, "840B")
    embedding = load_glove_embedding(50, "6B")
    runner = Runner(embedding=embedding,
                    log_path=args.log_path,
                    dataset=args.dataset)
    for model_class in [mz.models.ConvKNRM, mz.models.KNRM]:
        for train_ratio in [0.25, 0.5, .75, 1]:
            print("Prepare for Model %s" % model_class)
            runner.prepare_model(model_class, train_ratio=train_ratio)
            print(">>>>>>> Next Phase: train ratio = %.2f" % train_ratio)
            runner.run()


if __name__ == "__main__":
    main()
