from pathlib import Path

from torch.utils import data

from neural_ranking.data_loader.msmacro import MsMacroDocTriplesDataset, MsMacroDocReRankDataset


def test_masmacro():
    ms_macro_path = Path("~/ir/collection/ms-macro/document").expanduser()
    dataset = MsMacroDocReRankDataset(ms_macro_path).expanduser()

    loader = data.DataLoader(dataset, batch_size=8,
                             collate_fn=MsMacroDocReRankDataset.Collater("bert-base-uncased"),
                             num_workers=2)
