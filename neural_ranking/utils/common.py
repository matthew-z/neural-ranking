import random

from sklearn.model_selection import KFold


def slice_datapack_by_left_ids(datapack, left_ids, verbose=0):
    r = datapack.relation
    d = datapack.copy()

    if not left_ids:
        return d

    d.relation = r[r["id_left"].isin(left_ids)].reset_index(drop=1)
    original_index = set(d._left.index)

    if verbose:
        for idx in left_ids:
            if idx not in original_index:
                print("topic %d not found in datapack" % idx)

    d._left = d._left.reindex(left_ids)
    valid_doc_ids = sorted(list(set(d.relation.id_right)))
    d._right = d.right.reindex(valid_doc_ids)
    return d


def get_kfold_topics(pack, k=5, shuffle=False, seed=None):
    topics = pack.left.index.unique().tolist()
    kfold = KFold(k, shuffle=shuffle, random_state=seed)
    return kfold.split(topics)


def split_topics(pack, valid_size=0.2, test_size=0.2, shuffle=True, seed=None):
    topics = pack.left.index.unique().tolist()

    num_train = int(len(topics) * (1 - valid_size - test_size))
    num_valid = int(len(topics) * (valid_size))

    if shuffle:
        random.seed(seed)
        random.shuffle(topics)

    train_topics = topics[:num_train]
    if test_size > 0:
        valid_topics = topics[num_train:num_train + num_valid]
        test_topics = topics[num_train + num_valid:]
        return train_topics, valid_topics, test_topics

    valid_topics = topics[num_train:]
    return train_topics, valid_topics

    #
    # train_pack = slice_datapack_by_left_ids(pack, train_topics)
    # valid_pack = slice_datapack_by_left_ids(pack, valid_topics)
    # test_pack = slice_datapack_by_left_ids(pack, test_topics)
    #
    # return train_pack, valid_pack, test_pack
