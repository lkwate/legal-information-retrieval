from typing import Dict
import click
import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def score(data: pd.DataFrame, doc_mapping: Dict[int, set], k: int):
    results = []
    for _, row in data.iterrows():
        doc_id = row["id"]
        hits = set(map(lambda item: int(item[3:], str(row["hits"]).split(".")[:k])))
        match = len(hits & doc_mapping[doc_id]) / len(doc_mapping[doc_id])
        results.append(match)

    return np.mean(results)

@click.command()
@click.argument("input_result_file", type=click.Path(exists=True))
@click.argument("mapping", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("model", type=str)
def main(input_result_file: str, mapping: str, output_filepath: str, model: str):

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
    
    data = pd.read_csv(input_result_file)
    results = []
    for k in tqdm(range(1, 51)):
        match = score(data, doc_mapping, k)
        results.append(f"{model},{k},{match}{os.linesep}")
    
    open(output_filepath, "w").writelines(results)
if __name__ == "__main__":
    main()