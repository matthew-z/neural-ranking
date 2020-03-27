from itertools import chain

import numpy as np

import matchzoo as mz
from matchzoo.trainers import Trainer


def dict_mean(inputs):
    result = {}
    for i, d in enumerate(inputs):
        for k, v in d.items():
            if i != 0 and k not in result:
                raise ValueError("Expect same keys in the dict list, "
                                 "but encountered different keys in inputs")

            result.setdefault(k, [])
            result[k].append(v)

    for k, v in result.items():
        result[k] = float(np.mean(v))

    return result


def folds_to_kfolds(folds):
    for test_id in range(len(folds)):
        train_fold = folds[:test_id] + folds[test_id + 1:]
        test_fold = folds[test_id]
        yield list(chain(*train_fold)), test_fold


class ReRankTrainer(Trainer):
    def evaluate(
            self,
            dataloader):
        result = super().evaluate(dataloader)
        return result


def get_ranking_task(loss=None):
    ranking_task = mz.tasks.Ranking(losses=loss or mz.losses.RankHingeLoss())
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=20),
        mz.metrics.Precision(k=30),
    ]
    return ranking_task

