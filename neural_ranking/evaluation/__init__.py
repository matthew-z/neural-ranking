from typing import List

import keras
import matchzoo as mz
import numpy as np
from matchzoo.engine.base_model import BaseModel

class ReRankDataPack(object):
    def __init__(self, datapacks):
        self._datapacks = datapacks



def reranking_evalaute(unpacked_test_packs,
                       model: BaseModel,
                       batch_size: int = 128):
    results = {}
    for x, y  in unpacked_test_packs:
        result = model.evaluate(x, y, batch_size)
        for metric, score in result.items():
            results.setdefault(metric, [])
            results[metric].append(score)

    for metric, scores in results.items():
        results[metric] = np.mean(scores)
    return results

class EvaluateRankingMetrics(keras.callbacks.Callback):
    def __init__(
            self,
            model: 'BaseModel',
            test_packs: List[mz.DataPack],
            once_every: int = 1,
            batch_size: int = 128,
            model_save_path: str = None,
            verbose=1
    ):
        """Initializer."""
        super().__init__()
        self._model = model
        self._test_packs = [pack.unpack() for pack in test_packs]
        self._valid_steps = once_every
        self._batch_size = batch_size
        self._model_save_path = model_save_path
        self._verbose = verbose

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Called at the end of en epoch.

        :param epoch: integer, index of epoch.
        :param logs: dictionary of logs.
        :return: dictionary of logs.
        """
        if (epoch + 1) % self._valid_steps == 0:
            val_logs = reranking_evalaute(self._test_packs, self._model, self._batch_size)
            if self._verbose:
                print('Validation: ' + ' - '.join(
                    f'{k}: {v}' for k, v in val_logs.items()))
            for k, v in val_logs.items():
                logs[str(k)] = v
            if self._model_save_path:
                curr_path = self._model_save_path + str('%d/' % (epoch + 1))
                self._model.save(curr_path)
