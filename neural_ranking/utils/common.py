def split_datapack(pack, test_size=0.25):
    num_train = int(len(pack) * 0.8)
    pack.shuffle(inplace=True)
    train = pack[:num_train]
    test = pack[num_train:]

    return train, test