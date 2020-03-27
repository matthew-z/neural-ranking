import random

import numpy as np

from matchzoo.engine.base_callback import BaseCallback


class InsertQueryToDoc(BaseCallback):
    def __init__(self,
                 insert_mode="random",
                 ignore_positive=True,
                 positive_threshold=1,
                 ratio=1,
                 max_length=None
                 ):
        self.insert_mode = insert_mode
        self.ratio = ratio
        self.ignore_positive = ignore_positive
        self.positive_threshold = positive_threshold
        self.max_length = max_length

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        assert len(x["text_left"]) == len(x["text_right"])

        if x["text_right"].dtype != np.object:
            x["text_right"] = list(x["text_right"])

        for i in range(len(x["text_left"])):
            if self.ignore_positive and y[i] >= self.positive_threshold:
                continue

            if self.ratio < 1 and random.uniform(0, 1) > self.ratio:
                continue

            query = list(x["text_left"][i])
            doc = list(x["text_right"][i])
            if self.insert_mode == "pre":
                insert_pos = 0
            elif self.insert_mode == "random":
                insert_pos = random.choice(range(min(x["length_right"][i], self.max_length - len(query))))
            else:
                insert_pos = 0

            new_doc = doc[:insert_pos] + query + doc[insert_pos:]
            if self.max_length:
                new_doc = new_doc[:self.max_length]

            x["text_right"][i] = new_doc
            assert x["length_right"][i] == len(doc)
            x["length_right"][i] = len(new_doc)
