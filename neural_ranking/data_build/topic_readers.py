import pandas as pd
import bs4


class TopicsBuilder(object):
    def __init__(self):
        self.queries = []
        self.qids = []

    def add(self, query, qid=None):
        self.queries.append(query)
        self.qids.append(qid or len(self.qids))

    def to_df(self):
        return pd.DataFrame({"text_left":self.queries, "id_left":self.qids }, index=self.qids)


class TopicReader(object):
    def __call__(self, path):
        raise NotImplementedError

class TrecXmlTopicReader(TopicReader):
    def __call__(self, path, query_or_description="query"):
        topic_ids = []
        queries = []
        soup = bs4.BeautifulSoup(open(path), features="html.parser")
        topics = soup.find_all("topic")
        for topic in topics:
            qid = int(topic.attrs["number"])
            query = topic.find("query").text
            description = topic.find("description").text

            topic_ids.append(qid)
            queries.append(query)

        # use title only
        return pd.DataFrame({"text_left":queries, "id_left": topic_ids },
                            index=topic_ids)



class TrecRobustTopicReader(TopicReader):
    def __call__(self, path):
        topic_ids = []
        descriptions = []
        narratives = []
        queries = []

        with open(path) as fin:

            in_descr = False
            in_narr = False
            next_line_title =False

            descr_lines = []
            narr_lines = []

            for line in fin:

                if line.startswith("<desc"):
                    in_descr = True

                elif line.startswith("<narr"):
                    in_narr = True
                    in_descr = False
                    descriptions.append(" ".join(descr_lines).strip())
                    descr_lines = []

                elif in_descr:
                    descr_lines.append(line.strip())

                elif in_narr:
                    if line.startswith("</top>"):
                        in_narr = False
                        narratives.append(" ".join(narr_lines).strip())
                        narr_lines = []
                    else:
                        narr_lines.append(line.strip())

                elif line.startswith("<num>"):
                    topic_id = int((line.strip().split(":")[-1]).strip())
                    topic_ids.append(topic_id)

                elif line.startswith("<title>"):
                    if line.strip().endswith("<title>"):
                        next_line_title = True
                    else:
                        query = line.replace("<title>", "").strip()
                        queries.append(query)

                elif next_line_title:
                    queries.append(line.strip())
                    next_line_title = False

        # use title only
        return pd.DataFrame({"text_left":queries, "id_left": topic_ids }, index=topic_ids)

class NtcirTopicReader(TopicReader):
    def __call__(self, path):
        with open(path) as fin:
            soup = bs4.BeautifulSoup(fin, features="html.parser")
        builder = TopicsBuilder()

        raw_queries = soup.find_all("query")

        for q in raw_queries:
            qid = int(q.find("qid").text)
            query = q.find("content").text

            builder.add(query, qid)

        return builder.to_df()