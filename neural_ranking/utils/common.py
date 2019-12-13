import random
from sklearn.model_selection import KFold

def slice_datapack_by_left_ids(datapack, left_ids):
    r = datapack.relation
    d = datapack.copy()
    d.relation = r[r["id_left"].isin(left_ids)].reset_index(drop=1)
    d._left = d._left.loc[left_ids]

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
    valid_topics = topics[num_train:num_train + num_valid]
    test_topics = topics[num_train + num_valid:]

    return train_topics, valid_topics, test_topics

    #
    # train_pack = slice_datapack_by_left_ids(pack, train_topics)
    # valid_pack = slice_datapack_by_left_ids(pack, valid_topics)
    # test_pack = slice_datapack_by_left_ids(pack, test_topics)
    #
    # return train_pack, valid_pack, test_pack
