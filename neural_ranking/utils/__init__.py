import os
import dill
import matchzoo as mz
from matchzoo.preprocessors.units import WordHashing


def is_dssm_preprocessor(preprocessor):
    return isinstance(preprocessor, (mz.preprocessors.CDSSMPreprocessor, mz.preprocessors.DSSMPreprocessor))


def data_unpack(preprocessor, datapack):
    if not is_dssm_preprocessor(preprocessor):
        return datapack.unpack()
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    hashing = WordHashing(term_index=term_index)
    hashing.__name__ = "word_hashing"
    return datapack.apply_on_text(hashing.transform, verbose=False, inplace=False).unpack()


def load_rerank_data(path):
    packs = dill.load(open(os.path.join(path, "rerank.dill"), "rb"))
    return packs


def get_rerank_packs(rerank_packs, test_pack):
    test_left_ids = set(test_pack.left.index)
    test_rerank_packs = []
    for pack in rerank_packs:
        if pack.left.index[0] in test_left_ids:
            test_rerank_packs.append(pack)
    return test_rerank_packs
