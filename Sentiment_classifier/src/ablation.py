"""Ablation and hyperparameter sweep utilities."""
import itertools
import pandas as pd
import os

def run_grid(train_fn, grid_params, out_csv='ablation_results.csv'):
    """Run a grid of experiments.

    train_fn should be a callable accepting a dictionary of params and returning a dict of results.
    grid_params is a dict of parameter: list_of_values.
    """
    keys = list(grid_params.keys())
    rows = []
    for vals in itertools.product(*grid_params.values()):
        params = dict(zip(keys, vals))
        print('Running config:', params)
        try:
            res = train_fn(params)
            row = {**params, **res}
        except Exception as e:
            row = {**params, 'error': str(e)}
        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('Ablation helper loaded')
