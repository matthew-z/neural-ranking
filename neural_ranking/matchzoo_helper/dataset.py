from matchzoo.dataloader import Dataset
import matchzoo as mz
from matchzoo.engine.base_callback import BaseCallback

from typing import List


class ReRankDataset(Dataset):
    def __init__(self, data_pack: mz.DataPack, callbacks:List[BaseCallback], ):
        super().__init__(data_pack, callbacks=callbacks, mode="point")