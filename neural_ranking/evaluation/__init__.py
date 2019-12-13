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
