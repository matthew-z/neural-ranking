from pathlib import Path
import matchzoo as mz


_glove_6B_embedding_url = "http://nlp.stanford.edu/data/glove.6B.zip"
_glove_840B_embedding_url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"


def load_glove_embedding(dimension: int = 50, size="6B") -> mz.embedding.Embedding:
    """
    Return the pretrained glove embedding.

    :param dimension: the size of embedding dimension, the value can only be
        50, 100, or 300.
    :return: The :class:`mz.embedding.Embedding` object.
    """
    file_name = 'glove.{}.{}d.txt'.format(size, dimension)
    file_path = (Path(mz.USER_DATA_DIR) / 'glove').joinpath(file_name)

    if not file_path.exists():
        if size=="6B":
            url = _glove_6B_embedding_url
        elif size == "840B":
            url = _glove_840B_embedding_url
        else:
            raise ValueError("Incorrect Size for GloVe: %d" % size)

        mz.utils.get_file('glove_embedding',
                                        url,
                                        extract=True,
                                        cache_dir=mz.USER_DATA_DIR,
                                        cache_subdir='glove')

    return mz.embedding.load_from_file(file_path=str(file_path), mode='glove')

