from pathlib import Path

import bs4
import pandas as pd

import matchzoo as mz
from neural_ranking.data_build.topic_readers import TrecXmlTopicReader

class AsrCollection(object):
    def __init__(self, folder_path, topic_path):
        self.folder_path = Path(folder_path)
        self.topic_path = Path(topic_path)
        self.topic_df = TrecXmlTopicReader()(topic_path)

        with open(self.folder_path.joinpath("documents.rel")) as f:
            self.qrel_df = pd.read_csv(f, names=["id_left", "intent", "id_right", "rel"], index_col="id_right",
                                       dtype={"id_left": "int32", "rel": "int32"}, delimiter=" ")
            self.qrel_df.index = map(lambda s:s.replace("EPOCH", "ROUND"), self.qrel_df.index)
            self.qrel_df = self.qrel_df[["rel"]]

        with open(self.folder_path.joinpath("documents.ks")) as f:
            self.ks_df = pd.read_csv(f, names=["id_left", "intent", "id_right", "ks"], index_col="id_right",
                                     dtype={"id_left": "int32", "ks": "int32"}, delimiter="\s", engine="python")
            self.ks_df = self.ks_df[["ks"]]

        with open(self.folder_path.joinpath("documents.trectext")) as f:
            soup = bs4.BeautifulSoup(f, features="html.parser")
            texts = []
            round_nums = []
            query_ids = []
            author_ids = []
            doc_ids = []

            for doc in soup.find_all("doc"):
                doc_id = doc.find_next("docno").text
                _, round, query_id, author_id = doc_id.split("-")
                query_id = int(query_id)
                text = doc.find_next("text").text.strip()
                doc_ids.append(doc_id)
                texts.append(text)
                round_nums.append(round)
                query_ids.append(query_id)
                author_ids.append(author_id)

            self.df = pd.DataFrame({
                "id_left": query_ids,
                "id_right": doc_ids,
                "round_number": round_nums,
                "author_id": author_ids,
                "text_right": texts
            }, index=doc_ids)

            len1 = len(self.df)
            self.df = self.df.join(self.qrel_df, how="inner", on="id_right").join(self.ks_df, how="inner",
                                                                                  on="id_right").join(
                self.topic_df[["text_left"]], on="id_left", how="inner")
            len2 = len(self.df)
            assert (len1 == len2)

        tokenizer = mz.preprocessors.units.Tokenize()
        self._terms = []
        self.df["tokenized_text"] = self.df["text_right"].apply(lambda t: tokenizer.transform(t))
        self.df["tokenized_text"].apply(self._terms.extend)
        self.df["label"] = self.df["rel"]
        self._terms = set(self._terms)
        self.data_pack = mz.data_pack.pack(self.df[["id_left", "id_right", "label", "text_left", "text_right"]])

    @property
    def terms(self):
        return self._terms


if __name__ == "__main__":
    asrc = AsrCollection("./built_data/asrc", "./built_data/trec_web.1-200.asrc/topics")
    print(len(asrc.df))
    print(asrc.data_pack.frame())
