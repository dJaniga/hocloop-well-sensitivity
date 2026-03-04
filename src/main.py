import logging

import pandas as pd

from sensitivity_toolbox import (
    spearman_sensitivity,
    sobol_sensitivity,
    morris_sensitivity,
    rf_permutation_sensitivity,
)
from utils import setup_logging, save_sensitivity_results

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    params = pd.read_csv("data_source/parameters.csv")
    results = pd.read_csv("data_source/well_results.csv").drop(columns=["index"])

    w_positive_mask = results["w_out"] > 0

    params_f = params.loc[w_positive_mask]
    results_f = results.loc[w_positive_mask]

    spearman_sensitivity_results = spearman_sensitivity(params_f, results_f)
    sobol_sensitivity_results = sobol_sensitivity(
        params, results
    )  # must be aligned with params as data was produced for Sobol method
    moris_sensitivity_results = morris_sensitivity(
        params, results
    )  # must be aligned with params as data was produced for Sobol method
    rf_permutation_sensitivity_results = rf_permutation_sensitivity(params, results)

    save_sensitivity_results(
        spearman_results=spearman_sensitivity_results,
        sobol_results=sobol_sensitivity_results,
        morris_results=moris_sensitivity_results,
        rf_results=rf_permutation_sensitivity_results,
    )


if __name__ == "__main__":
    main()
