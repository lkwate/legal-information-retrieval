import click
import pandas as pd
from .entailment_searcher import EntailmentSearcher
import os
import json


@click.command()
@click.argument("model", type=str)
@click.argument("dense_index", type=click.Path(exists=True))
@click.argument("sparse_index", type=click.Path(exists=True))
@click.argument("documents", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--hidden_dim", type=int, default=768)
@click.option("--top_k", type=int, default=5)
@click.option("--device", type=str, default="cpu")
def main(
    model: str,
    dense_index: str,
    sparse_index: str,
    documents: str,
    output_filepath,
    hidden_dim: int,
    top_k: int,
    device: str,
):
    searcher = EntailmentSearcher(
        model,
        dense_index,
        sparse_index,
        documents,
        hidden_dim=hidden_dim,
        top_k=top_k,
        device=device,
    )
    output = []
    for filename in os.listdir(documents):
        input_filepath = os.path.join(documents, filename)
        query = json.loads(open(input_filepath).read())["contents"]
        result = searcher(query)
        output.append(result)
    
    with open(output_filepath, "w") as f:
        json.dump(output, f)

if __name__ == "__main__":
    main()
