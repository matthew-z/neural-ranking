import numpy as np

from matchzoo.engine.base_callback import BaseCallback
import random

class InsertQueryToDoc(BaseCallback):
    def __init__(self,
                 termIndex,
                 insert_mode="pre",
                 ignore_positive=True,
                 positive_threshold=1,
                 ratio=1):
        self.termIndex = termIndex
        self.insert_mode = insert_mode
        self.ratio = ratio
        self.ignore_positive = ignore_positive
        self.positive_threshold = positive_threshold

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        assert len(x["text_left"]) == len(x["text_right"])
        space = self.termIndex[" "]

        if x["text_right"].dtype != np.object:
            x["text_right"] = list(x["text_right"])

        for i in range(len(x["text_left"])):
            if self.ignore_positive and y[i] >= self.positive_threshold:
                continue

            if self.ratio < 1 and random.uniform(0, 1) < self.ratio:
                continue

            query = [space] + list(x["text_left"][i]) + [space]
            doc = list(x["text_right"][i])
            if self.insert_mode == "pre":
                insert_pos = 0
            else:
                insert_pos = 0
            new_doc = doc[:insert_pos] + query + doc[insert_pos:]
            x["text_right"][i] = new_doc
            assert x["length_right"][i] == len(doc)
            x["length_right"][i] = len(new_doc)
