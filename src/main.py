import logging

import pandas as pd

from methods import (
    spearman_sensitivity,
    morris_sensitivity,
    sobol_sensitivity,
    rf_permutation_sensitivity,
    pce_sensitivity,
    hdmr_sensitivity,
)

from utils import save_sensitivity_results, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:

    setup_logging()

    params = pd.read_csv("data_source/parameters.csv")
    results = pd.read_csv("data_source/well_results.csv").drop(columns=["index"])

    save_sensitivity_results(**{"spearman": spearman_sensitivity(params, results)})
    save_sensitivity_results(**{"morris": morris_sensitivity(params, results)})
    save_sensitivity_results(**{"sobol": sobol_sensitivity(params, results)})
    save_sensitivity_results(**{"rf_perm": rf_permutation_sensitivity(params, results)})
    save_sensitivity_results(**{"pce": pce_sensitivity(params, results, order=2)})
    save_sensitivity_results(**{"hdmr": hdmr_sensitivity(params, results, order=2)})


if __name__ == "__main__":
    main()
