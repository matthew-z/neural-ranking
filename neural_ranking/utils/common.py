import random


def slice_datapack_by_left_ids(datapack, left_ids):
    r = datapack.relation
    d = datapack.copy()
    d.relation = r[r["id_left"].isin(left_ids)].reset_index(drop=1)
    d._left = d._left.loc[left_ids]
    return d


def split_datapack(pack, valid_size=0.2, test_size=0.2, shuffle=True):
    topics = pack.left.index.unique().tolist()

    num_train = int(len(topics) * (1 - valid_size - test_size))
    num_valid = int(len(topics) * (valid_size))

    if shuffle:
        random.shuffle(topics)

    train_topics = topics[:num_train]
    valid_topics = topics[num_train:num_train + num_valid]
    test_topics = topics[num_train + num_valid:]

    train_pack = slice_datapack_by_left_ids(pack, train_topics)
    valid_pack = slice_datapack_by_left_ids(pack, valid_topics)
    test_pack = slice_datapack_by_left_ids(pack, test_topics)

    return train_pack, valid_pack, test_pack
