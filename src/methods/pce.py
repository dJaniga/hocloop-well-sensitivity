import itertools
import logging
import warnings

import chaospy as cp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*where.*unitialized memory.*")


def pce_sensitivity(params_f: pd.DataFrame, results_f: pd.DataFrame, order: int = 2):
    """
    Perform Polynomial Chaos Expansion (PCE) sensitivity analysis on the given parameters and results.
    """

    logger.info("Starting PCE sensitivity analysis (order=%s)", order)

    feature_names = list(params_f.columns)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(params_f)

    logger.info(
        "Samples=%s | Parameters=%s | Outputs=%s",
        X.shape[0],
        X.shape[1],
        len(results_f.columns),
    )

    distributions = [cp.Uniform(0, 1) for _ in feature_names]
    joint_dist = cp.J(*distributions)

    poly_expansion = cp.expansion.stieltjes(order, joint_dist)

    results = {}

    for out in results_f.columns:
        logger.info("Processing output: %s", out)

        y = results_f[out].values

        coeffs = cp.fit_regression(poly_expansion, X.T, y)

        S1 = cp.Sens_m(coeffs, joint_dist)
        ST = cp.Sens_t(coeffs, joint_dist)

        main_df = pd.DataFrame({"S1": S1, "ST": ST}, index=feature_names).sort_values(
            "ST", ascending=False
        )

        interaction_results = {}

        # compute higher order interactions dynamically
        for k in range(2, order + 1):
            sens_func = getattr(cp, f"Sens_m{k}", None)

            if sens_func is None:
                continue

            S_k = sens_func(coeffs, joint_dist)

            rows = []

            for comb in itertools.combinations(range(len(feature_names)), k):
                name = tuple(feature_names[i] for i in comb)

                value = S_k[comb]

                rows.append({"parameters": name, f"S{k}": value})

            df_k = pd.DataFrame(rows).sort_values(f"S{k}", ascending=False)

            interaction_results[f"S{k}"] = df_k

        results[out] = {
            "main_effects": main_df,
            "interactions": interaction_results,
        }

        logger.info("Top parameter for %s: %s", out, main_df.index[0])

    logger.info("PCE sensitivity analysis completed")

    return results
