import pandas as pd

class TopicReader(object):
    def __call__(self, path):
        raise NotImplementedError

class TrecTopicReader(TopicReader):
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
        return pd.DataFrame({"text_left":queries }, index=topic_ids)
