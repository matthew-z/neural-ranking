def split_datapack(pack, valid_size=0.2, test_size=0.2):
    num_train = int(len(pack) * (1 - valid_size - test_size))
    num_valid = int(len(pack) * (valid_size))

    pack.shuffle(inplace=True)

    train = pack[:num_train]
    valid = pack[num_train:num_train+num_valid]
    test = pack[num_train+num_valid:]

    return train, valid, test
