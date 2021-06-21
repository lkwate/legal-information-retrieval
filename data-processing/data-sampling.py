import pandas as pd
import numpy as np
import os
import click


@click.command()
@click.argument('input_path_data', type=click.Path(exists=True))
@click.argument('output_path_data', type=click.Path(exists=True))
@click.argument('label', type=str, default='label')
@click.argument('seed', type=int, default=42)
@click.argument('num_experiment', type=int, default=10)
@click.argument('factor', type=int, default=.05)
def main(
    input_path_data: str,
    output_path_data: str,
    label: str,
    seed: int = 42,
    num_experiment: int = 10,
    factor: float = 0.05,
) -> None:
    np.random.seed(seed)
    data = pd.read_csv(input_path_data)

    class_one = data[data[label] == 1]
    class_two = data[data[label] == 0]
    size_class_two = len(class_one) + int(len(class_two) * factor)

    for i in range(1, num_experiment + 1):
        sampled_data = class_two.sample(size_class_two)
        output_file_path = os.path.join(output_path_data, f"experiment_{i}")
        sampled_data = pd.concat([class_one, sampled_data], axis=0)
        sampled_data.to_csv(output_file_path, index=False)

if __name__ == '__main__':
    main()