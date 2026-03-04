import logging

import pandas as pd

from pipeline.api import run_sensitivity_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    params = pd.read_csv("data_source/parameters.csv")
    results = pd.read_csv("data_source/well_results.csv").drop(columns=["index"])

    run_sensitivity_pipeline(params, results, n_jobs=4)


if __name__ == "__main__":
    main()
