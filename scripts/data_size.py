import os
import pickle
from pathlib import Path

import keras
import matchzoo as mz
import tqdm
from matchzoo.losses import RankHingeLoss
from matchzoo.preprocessors.units import WordHashing

from neural_ranking.embedding.glove import load_glove_embedding
from neural_ranking.evaluation import EvaluateRankingMetrics, reranking_evalaute
from neural_ranking.utils import is_dssm_preprocessor, load_rerank_data, get_rerank_packs
from neural_ranking.utils.common import split_datapack

DATA_FOLDER = Path("~/ir/neural_ranking/built_data/").expanduser()


class Runner(object):
    def __init__(self, dataset="robust04", embedding=None, log_path="log"):
        self.dataset = dataset
        datapath = DATA_FOLDER.joinpath(dataset)
        pack = mz.load_data_pack(datapath).shuffle()
        rerank_packs = load_rerank_data(datapath)
        self.train_pack, valid_pack, test_pack = split_datapack(pack)
        self.valid_rerank_packs = get_rerank_packs(rerank_packs, valid_pack)
        self.test_rerank_packs = get_rerank_packs(rerank_packs, test_pack)


        if embedding is None:
            self.raw_embedding = load_glove_embedding(dimension=50, size="6B")
        else:
            self.raw_embedding = embedding

        self.log_path = Path(log_path).absolute()


    def prepare(self, model_cls, preprocessor, params):
        self.preprocessor = preprocessor
        self.model_cls = model_cls
        self.params = params
        self.train_pack_processed = preprocessor.fit_transform(self.train_pack)
        self.valid_packs_processed = [preprocessor.transform(pack, verbose=0) for pack in
                                      tqdm.tqdm(self.valid_rerank_packs, desc="processing validation set ")]
        self.test_packs_processed = [preprocessor.transform(pack, verbose=0) for pack in
                                     tqdm.tqdm(self.test_rerank_packs, desc="processing test set")]
        self.embedding_matrix = self.raw_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
        self.reset_params()

    def reset_params(self):
        print(">>> context: %s" % self.preprocessor.context)
        self.model = self.model_cls()
        self.model.params.update(self.preprocessor.context)
        self.model.params.update(self.params)
        self.model.guess_and_fill_missing_params()
        self.model.build()
        self.model.compile()

        if not is_dssm_preprocessor(self.preprocessor):
            self.model.load_embedding_matrix(self.embedding_matrix)

    def run(self,
            train_ratio=1.0,
            epochs=10,
            train_gen_mode='pair',
            train_gen_num_dup=5, train_gen_num_neg=1,
            batch_size=64):

        if is_dssm_preprocessor(self.preprocessor):
            term_index = self.preprocessor.context['vocab_unit'].state['term_index']
            hashing = WordHashing(term_index=term_index)
            hashing.__name__ = "word_hashing"
            callbacks = [
                mz.data_generator.callbacks.LambdaCallback(
                    on_batch_data_pack=lambda pack: pack.apply_on_text(hashing.transform, verbose=False, inplace=True))]

        else:
            callbacks = None

        train_size = int(len(self.train_pack_processed) * train_ratio)
        train_generator = mz.DataGenerator(
            self.train_pack_processed[:train_size].copy(),
            mode=train_gen_mode,
            num_dup=train_gen_num_dup,
            num_neg=train_gen_num_neg,
            batch_size=batch_size,
            callbacks=callbacks
        )

        valid_evaluate = EvaluateRankingMetrics(self.model, test_packs=self.valid_packs_processed, batch_size=128)
        early_stopping = keras.callbacks.EarlyStopping(monitor='normalized_discounted_cumulative_gain@5(0.0)',
                                                       min_delta=0, patience=5, verbose=1, mode='max',
                                                       baseline=None, restore_best_weights=True)

        history = self.model.fit_generator(train_generator, epochs=epochs,
                                           workers=30, use_multiprocessing=True,
                                           callbacks=[valid_evaluate, early_stopping])

        # evaluate
        evaluate_test = reranking_evalaute([pack.unpack() for pack in self.test_packs_processed], self.model)
        model_name = self.model.__class__.__name__.split(".")[-1]
        history_name = "%s_%s_%.2f.result.pkl" % (model_name, self.dataset, train_ratio)

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


def conv_knrm(runner):
    model_cls = mz.models.ConvKNRM
    preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=5, fixed_length_right=512,
                                                      remove_stop_words=True, multiprocessing=True)
    params = {}
    params['task'] = get_ranking_task()
    params['embedding_output_dim'] = runner.raw_embedding.output_dim
    params['embedding_trainable'] = True
    params['filters'] = 128
    params['conv_activation_func'] = 'tanh'
    params['max_ngram'] = 3
    params['use_crossmatch'] = True
    params['kernel_num'] = 11
    params['sigma'] = 0.1
    params['exact_sigma'] = 0.001
    params['optimizer'] = 'adadelta'

    return model_cls, preprocessor, params


def knrm(runner):
    model_cls = mz.models.KNRM
    preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=5, fixed_length_right=512,
                                                      remove_stop_words=True, multiprocessing=True)
    params = {}
    params['task'] = get_ranking_task()
    params['embedding_output_dim'] = runner.raw_embedding.output_dim
    params['embedding_trainable'] = True
    params['kernel_num'] = 21
    params['sigma'] = 0.1
    params['exact_sigma'] = 0.001
    params['optimizer'] = 'adadelta'

    return model_cls, preprocessor, params


def dssm(runner):
    model_cls = mz.models.DSSM
    preprocessor = mz.preprocessors.DSSMPreprocessor(with_word_hashing=False)
    params = {}
    params['task'] = get_ranking_task(mz.losses.RankCrossEntropyLoss(num_neg=1))
    params['embedding_output_dim'] = runner.raw_embedding.output_dim
    params['mlp_num_layers'] = 3
    params['mlp_num_units'] = 300
    params['mlp_num_fan_out'] = 128
    params['mlp_activation_func'] = 'relu'

    return model_cls, preprocessor, params


def cdssm(runner):
    model_cls = mz.models.CDSSM
    preprocessor = mz.preprocessors.CDSSMPreprocessor(fixed_length_left=5, fixed_length_right=512,
                                                      with_word_hashing=False)
    params = {}
    params['embedding_output_dim'] = runner.raw_embedding.output_dim
    params['task'] = get_ranking_task()
    params['filters'] = 64
    params['kernel_size'] = 3
    params['strides'] = 1
    params['padding'] = 'same'
    params['conv_activation_func'] = 'tanh'
    params['w_initializer'] = 'glorot_normal'
    params['b_initializer'] = 'zeros'
    params['mlp_num_layers'] = 1
    params['mlp_num_units'] = 64
    params['mlp_num_fan_out'] = 64
    params['mlp_activation_func'] = 'tanh'
    params['dropout_rate'] = 0.8
    params['optimizer'] = 'adadelta'

    return model_cls, preprocessor, params


def duet(runner):
    model_cls = mz.models.DUET
    preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=5, fixed_length_right=512,
                                                      remove_stop_words=True, multiprocessing=True)
    params = {}
    params['task'] = get_ranking_task()
    params['embedding_output_dim'] = runner.raw_embedding.output_dim
    params['lm_filters'] = 32
    params['lm_hidden_sizes'] = [32]
    params['dm_filters'] = 32
    params['dm_kernel_size'] = 3
    params['dm_d_mpool'] = 4
    params['dm_hidden_sizes'] = [32]
    params['dropout_rate'] = 0.3
    params['optimizer'] = 'adagrad'

    return model_cls, preprocessor, params


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="robust04")
    parser.add_argument("--log-path", type=path, default="data_size_log")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    embedding = load_glove_embedding(300, "840B")
    # embedding = load_glove_embedding(50, "6B")
    for model_fn in [duet, knrm, conv_knrm]:
        print(model_fn)
        runner = Runner(embedding=embedding, log_path=args.log_path, dataset=args.dataset)
        runner.prepare()
        for train_size in [0.25, 0.5, .75, 1]:
            print(">>>>>>> Next Phase: train size = %.2f" % train_size)
            runner.run(train_ratio=train_size)
            runner.reset_params()


if __name__ == "__main__":
    main()
