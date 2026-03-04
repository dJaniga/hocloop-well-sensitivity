import logging

import pandas as pd
from SALib.analyze import sobol

from pipeline import register_sensitivity_method

logger = logging.getLogger(__name__)


@register_sensitivity_method("sobol")
def sobol_sensitivity(params_f: pd.DataFrame, results_f: pd.DataFrame):
    """
    Compute Sobol sensitivity indices.

    Parameters
    ----------
    params_f : pd.DataFrame
        Filtered input parameters
    results_f : pd.DataFrame
        Filtered model outputs

    Returns
    -------
    dict[str, pd.DataFrame]
        Sobol indices for each output variable
    """

    logger.info("Computing Sobol sensitivity indices")

    problem = {
        "num_vars": params_f.shape[1],
        "names": list(params_f.columns),
        "bounds": [[params_f[c].min(), params_f[c].max()] for c in params_f.columns],
    }

    results = {}

    for out in results_f.columns:
        Y = results_f[out].values

        Si = sobol.analyze(problem, Y, calc_second_order=True)

        df = pd.DataFrame(
            {
                "S1": Si["S1"],
                "S1_conf": Si["S1_conf"],
                "ST": Si["ST"],
                "ST_conf": Si["ST_conf"],
            },
            index=problem["names"],
        )

        results[out] = df

    return results
