import matchzoo as mz
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split

from neural_ranking.embedding.glove import load_glove_embedding
from neural_ranking.utils.common import split_datapack

from pathlib import Path

DATA_FOLDER = Path("~/ir/neural_ranking/built_data/robust04/").expanduser()

class Runner(object):
    def __init__(self, dataset="robust04", embedding=None):
        self.pack = mz.load_data_pack(DATA_FOLDER.joinpath(dataset)).shuffle()
        if embedding is None:
            self.embedding = load_glove_embedding(dimension=50, size="6B")


