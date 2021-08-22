import click
from pyserini.dsearch import AutoQueryEncoder, SimpleDenseSearcher
from tqdm import tqdm
import json
import os
from loguru import logger
import numpy as np

@click.command()
@click.argument("model", type=click.Path())
@click.argument("index", type=click.Path(exists=True))
@click.argument("documents", type=click.Path(exists=True))
@click.argument("mapping", type=click.Path(exists=True))
@click.argument("top_k", type=int)
@click.argument("output_filepath", type=click.Path(exists=True))
@click.option("--separator", type=str, default=",")
def main(
    model: str,
    index: str,
    documents: str,
    mapping: str,
    top_k: int,
    output_filepath: str,
    separator: str
):
    # build mapping
    doc_mapping = {}
    for line in open(mapping).readlines():
        line = line.split(";")
        if len(line) == 1:
            continue
        doc_id = int(line[0])
        mapped_docs = set(map(int, line[1].split(",")))
        doc_mapping[doc_id] = mapped_docs
        
    # searcher
    encoder = AutoQueryEncoder(encoder_dir=model, pooling="mean", l2_norm=True)
    searcher = SimpleDenseSearcher.from_prebuilt_index(index, encoder)
    
    score = []
    for doc_id, mapped_docs in tqdm(doc_mapping.items()):
        input_filepath = os.path.join(documents, f"doc{doc_id}.json")
        contents = json.loads(open(input_filepath).read())["contents"]
        
        hits = searcher.search(contents)[:top_k]
        doc_ids = set(map(lambda item: item.docid, hits))
        matchs = doc_ids & mapped_docs
        score.append(len(matchs) / len(mapped_docs))
    
    score = np.mean(score)
    with open(output_filepath, "a") as f:
        line = separator.join([model, str(top_k), str(score)])
        logger.info(line)
        f.write(f"{line}{os.linesep}")

if __name__ == "__main__":
    main()
