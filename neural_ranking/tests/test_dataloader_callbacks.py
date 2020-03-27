import matchzoo as mz
from neural_ranking.data_build.callbacks import InsertQueryToDoc


def test_attach_query():
    pack = mz.datasets.toy.load_data()
    p = mz.models.MatchLSTM.get_default_preprocessor()
    pack_p  = p.fit_transform(pack)

    ic = InsertQueryToDoc(p.context["vocab_unit"].context["term_index"])
    dataset = mz.dataloader.Dataset(pack_p, callbacks=[ic,], batch_size=2, shuffle=False)
    dataloader = mz.dataloader.DataLoader(dataset, callback= mz.models.MatchLSTM.get_default_padding_callback())

    for x, y in dataloader:
        print(x["text_right"])
