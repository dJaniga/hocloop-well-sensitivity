import copy
import logging

import pandas as pd
from SALib.analyze import enhanced_hdmr


logger = logging.getLogger(__name__)


def hdmr_sensitivity(params_f: pd.DataFrame, results_f: pd.DataFrame, order=3):
    """
    Perform enhanced HDMR sensitivity analysis on the given parameters and results.
    """

    logger.info("Starting enhanced HDMR sensitivity analysis")

    feature_names = list(params_f.columns)

    X = params_f.to_numpy()

    n_samples, n_features = X.shape

    logger.info("Samples=%s | Parameters=%s", n_samples, n_features)

    problem = {
        "num_vars": n_features,
        "names": feature_names,
        "bounds": [[params_f[c].min(), params_f[c].max()] for c in feature_names],
    }

    results = {}

    for out in results_f.columns:
        logger.info("Processing output: %s", out)

        Y = results_f[out].to_numpy()

        problem_local = copy.deepcopy(problem)

        Si = enhanced_hdmr.analyze(
            problem=problem_local, X=X, Y=Y, max_order=order, print_to_console=False
        )

        df = pd.DataFrame(
            {"S": Si["S"], "ST": Si["ST"]}, index=problem_local["names"]
        ).sort_values("ST", ascending=False)

        results[out] = df

        logger.info("Top parameter: %s", df.index[0])

    logger.info("Enhanced HDMR analysis finished")

    return results
