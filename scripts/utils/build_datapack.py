import argparse
import shutil
import os
import json
from pathlib import Path

from neural_ranking.data.build_pack_from_qrels import TrecDataBuilder, NtcirDataBuilder, TrecXmlBuilder


def path(str):
    return Path(os.path.abspath((os.path.expanduser(str))))


def copy_qrel_and_topics(topics_path: Path, qrels_path: Path,
                         output_path: Path):
    shutil.copyfile(qrels_path, output_path.joinpath("qrels"))
    shutil.copy(topics_path, output_path.joinpath("topics"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--index", type=path)
    parser.add_argument("--qrel", type=path)
    parser.add_argument("--topic", type=path)
    parser.add_argument("--topic-format", type=str, default='trec',
                        choices=["trec", "ntcir", "xml"])
    parser.add_argument("--output", type=path,
                        default="./built_data/my-data-pack")
    parser.add_argument("--filtered-topics", type=path)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--hits", type=int, default=1000)
    args = parser.parse_args()


    if args.topic_format == "trec":
        builder = TrecDataBuilder(args.index, args.topic, args.qrel)
    elif args.topic_format == "ntcir":
        builder = NtcirDataBuilder(args.index, args.topic, args.qrel)
    elif args.topic_format == "xml":
        builder = TrecXmlBuilder(args.index, args.topic, args.qrel)
    else:
        raise ValueError

    filtered_topics = json.load(open(args.filtered_topics)) if args.filtered_topics else None
    builder.build_datapack(args.output, filtered_topics)
    if not args.no_rerank:
        builder.build_rerank_datapack(args.output, args.hits)
    copy_qrel_and_topics(args.topic, args.qrel, args.output)
