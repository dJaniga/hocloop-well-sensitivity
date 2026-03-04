import logging

import pandas as pd

logger = logging.getLogger(__name__)


def spearman_sensitivity(params: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Spearman sensitivity between model parameters and outputs.

    Parameters
    ----------
    params : pd.DataFrame
        Input parameters (each column = parameter)
    results : pd.DataFrame
        Model outputs (each column = result)

    Returns
    -------
    pd.DataFrame
        Spearman correlation matrix (parameters x outputs)
    """

    logger.info("Computing Spearman sensitivity indices")

    # ensure same number of samples
    if len(params) != len(results):
        raise ValueError("params and results must have the same number of rows")

    sensitivity = pd.DataFrame(index=params.columns, columns=results.columns)

    for r in results.columns:
        sensitivity[r] = params.corrwith(results[r], method="spearman")

    return sensitivity
