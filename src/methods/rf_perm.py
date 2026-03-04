import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from pipeline import register_sensitivity_method

logger = logging.getLogger(__name__)


@register_sensitivity_method("rf_perm")
def rf_permutation_sensitivity(
    params_f: pd.DataFrame,
    results_f: pd.DataFrame,
    n_estimators: int = 200,
    n_repeats: int = 10,
):
    """
    Random Forest + Permutation Sensitivity analysis.

    Returns
    -------
    dict[str, dict]
        For each output:
        - rf_importance
        - permutation_importance
    """

    logger.info("Starting RF + Permutation sensitivity analysis")

    X = params_f.values
    feature_names = params_f.columns

    n_samples, n_features = params_f.shape
    logger.info(
        "Samples=%s | Parameters=%s | Outputs=%s",
        n_samples,
        n_features,
        len(results_f.columns),
    )

    results = {}

    for out in results_f.columns:
        logger.info("Processing output: %s", out)

        y = results_f[out].values

        rf = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=1, random_state=42, verbose=0
        )

        rf.fit(X, y)

        logger.debug("Computing RF feature importance")

        rf_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(
            ascending=False
        )

        logger.debug("Computing permutation importance")

        perm = permutation_importance(
            rf, X, y, n_repeats=n_repeats, n_jobs=-1, random_state=42
        )

        perm_imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(
            ascending=False
        )

        results[out] = {"rf_importance": rf_imp, "permutation_importance": perm_imp}

        logger.info(
            "Finished %s | Top RF: %s | Top Permutation: %s",
            out,
            rf_imp.index[0],
            perm_imp.index[0],
        )

    logger.info("RF + Permutation sensitivity analysis completed")

    return results
