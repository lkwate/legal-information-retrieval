import click
from pyserini.search import SimpleSearcher
from tqdm import tqdm
import json
import os
from loguru import logger
import numpy as np
import pandas as pd


@click.command()
@click.argument("index", type=click.Path(exists=True))
@click.argument("documents", type=click.Path(exists=True))
@click.argument("mapping", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(
    index: str,
    documents: str,
    mapping: str,
    output_filepath: str,
):
    # build mapping
    doc_mapping = {}
    for line in open(mapping).readlines():
        line = line.rstrip()
        line = line.split(";")
        if line[1] == "":
            continue
        doc_id = int(line[0])
        mapped_docs = set(map(int, line[1].split(",")))
        doc_mapping[doc_id] = mapped_docs

    # searcher
    searcher = SimpleSearcher(index)
    score = []
    for doc_id, mapped_docs in tqdm(doc_mapping.items()):
        input_filepath = os.path.join(documents, f"doc{doc_id}.json")
        contents = json.loads(open(input_filepath).read())["contents"]

        hits = searcher.search(contents, k=50)
        doc_ids = ".".join(list(map(lambda item: str(item.docid), hits)))
        score.append((doc_id, doc_ids))

    data = pd.DataFrame(data=score, columns=["id", "hits"])
    data.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    main()
