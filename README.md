# Lev Kozlov - B20-RO

Email: l dot kozlov at innopolis dot university

## Report

The final report of this assignment can be found [here](reports/final_report.md).

## Requrements:

- I have tested the code on Ubuntu distribution with python 3.10
- Export of `conda` env can be found in [environment.yml](environment.yml)

## Structure:

See [PROBLEM.md](PROBLEM.md) for the task description.

- `data/` - folder with data, contains both `interim` and `raw` data
- `benchmark/` - python module in which all losses, models, dataset and other python code is stored
- `notebooks/` - jupyter notebooks with analysis
- `models/` - folder with saved states of models
- `reports/` - folder with final report and figures

## Usage:

1. Activate conda environment
2. Run `python3 benchmark/evaluate.py`. In the output you will see the results of evaluation of all models.
