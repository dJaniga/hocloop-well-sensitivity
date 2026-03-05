import logging

import pandas as pd
from SALib.analyze import sobol


logger = logging.getLogger(__name__)


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
    dict[str, dict[str, pd.DataFrame]]
        Sobol indices for each output variable, containing:
        - 'main_effects': DataFrame with S1 and ST indices
        - 'interactions': dict with 'second_order' DataFrame
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

        # Main effects (first-order and total)
        main_effects = pd.DataFrame(
            {
                "S1": Si["S1"],
                "S1_conf": Si["S1_conf"],
                "ST": Si["ST"],
                "ST_conf": Si["ST_conf"],
            },
            index=problem["names"],
        )

        # Second-order interactions
        s2_matrix = Si["S2"]
        s2_conf_matrix = Si["S2_conf"]

        # Create list of interaction pairs with their indices
        interactions_data = []
        for i in range(len(problem["names"])):
            for j in range(i + 1, len(problem["names"])):
                interactions_data.append({
                    "Parameter_1": problem["names"][i],
                    "Parameter_2": problem["names"][j],
                    "S2": s2_matrix[i, j],
                    "S2_conf": s2_conf_matrix[i, j],
                })

        second_order = pd.DataFrame(interactions_data)

        results[out] = {
            "main_effects": main_effects,
            "interactions": {"second_order": second_order}
        }

    return results