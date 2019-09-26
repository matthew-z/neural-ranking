import html2text
import bs4
import multiprocessing
import tqdm

class SimpleHTMLExtractor():
    def __init__(self):
        h = html2text.HTML2Text()
        h.ignore_emphasis = 1
        h.ignore_images = 1
        h.ignore_links = 1
        h.ignore_tables = 1
        self.handler = h

    def __call__(self, doc):
        return self.handler.handle(doc)


    def bulk_extract(self, docs):

        with multiprocessing.Pool() as p:
            r = list(tqdm.tqdm(p.imap(self, docs, chunksize=16), total=len(docs)))

        return r



class Robust04Extractor(SimpleHTMLExtractor):
    def __call__(self, doc):
        text =  super().__call__(doc)

        return text.split("[Text]")[-1]

