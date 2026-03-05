import logging

import pandas as pd
from SALib.analyze import morris

logger = logging.getLogger(__name__)


def morris_sensitivity(params_f: pd.DataFrame, results_f: pd.DataFrame):
    """
    Compute Morris sensitivity indices.

    Returns
    -------
    dict[str, pd.DataFrame]
        mu, mu_star and sigma for each output variable
    """

    logger.info("Computing Morris sensitivity indices")

    problem = {
        "num_vars": params_f.shape[1],
        "names": list(params_f.columns),
        "bounds": [[params_f[c].min(), params_f[c].max()] for c in params_f.columns],
    }

    X = params_f.values
    results = {}

    for out in results_f.columns:
        Y = results_f[out].values

        Si = morris.analyze(problem, X, Y, conf_level=0.95, print_to_console=False)

        df = pd.DataFrame(
            {"mu": Si["mu"], "mu_star": Si["mu_star"], "sigma": Si["sigma"]},
            index=problem["names"],
        )

        results[out] = df.sort_values("mu_star", ascending=False)

    return results
