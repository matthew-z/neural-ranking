import itertools
from pprint import pprint

from .rbo import rbo


def kendallTau(A, B):
    pairs = itertools.combinations(range(0, len(A)), 2)

    distance = 0
    pair_n = 0
    for x, y in pairs:
        pair_n += 1
        a = A[x] - A[y]
        b = B[x] - B[y]

        # if discordant (different signs)
        if (a * b < 0):
            distance += 1

    return distance / pair_n


class RobustnessMetric(object):
    def __call__(self, doc_list1, doc_list2):
        assert len(doc_list1) == len(doc_list2)
        assert len(doc_list1) >= 1
        return self.call(sorted(doc_list1, reverse=True), sorted(doc_list2, reverse=True))

    def call(self, doc_list1, doc_list2):
        raise NotImplementedError


class TopChange(RobustnessMetric):
    def call(self, doc_list1, doc_list2):
        return doc_list1[0][1] != doc_list2[0][1]


class KendallsDistance(RobustnessMetric):
    def call(self, doc_list1, doc_list2):
        author_ids_1 = [author_id for score, author_id in doc_list1]
        author_ids_2 = [author_id for score, author_id in doc_list2]

        assert sorted(author_ids_1) == sorted(author_ids_2)
        authors = sorted(author_ids_1)
        ranks1 = [author_ids_1.index(aid) for aid in authors]
        ranks2 = [author_ids_2.index(aid) for aid in authors]

        return kendallTau(ranks1, ranks2)


class RBO(RobustnessMetric):
    def __init__(self, mode="min"):
        self.mode = mode

    def call(self, doc_list1, doc_list2):
        author_ids_1 = [author_id for score, author_id in doc_list1]
        author_ids_2 = [author_id for score, author_id in doc_list2]
        res = rbo(author_ids_1, author_ids_2, p=0.7)
        if self.mode == "min":
            return res.min
        if self.mode == "avg":
            return (res.min + res.max) / 2
        else:
            return res.ext

def get_robustness_metrics():
    return {
        "TopChange": TopChange(),
        "KT": KendallsDistance(),
        "RBO_min": RBO("min"),
        "RBO_ext": RBO("ext"),
        "RBO_avg": RBO("avg")
    }
