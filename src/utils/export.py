import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_sensitivity_results(
    spearman_results: pd.DataFrame,
    sobol_results: dict,
    morris_results: dict,
    rf_results: dict,
    output_dir: str = "results",
) -> None:
    """
    Save all sensitivity analysis results to CSV and JSON files.

    Parameters
    ----------
    spearman_results : pd.DataFrame
        Spearman correlation results
    sobol_results : dict[str, pd.DataFrame]
        Sobol sensitivity indices for each output
    morris_results : dict[str, pd.DataFrame]
        Morris sensitivity indices for each output
    rf_results : dict[str, dict[str, pd.Series]]
        RF and permutation importance for each output
    output_dir : str
        Directory to save results to
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    logger.info("Saving sensitivity results to %s", output_dir)

    # Save Spearman results (DataFrame)
    spearman_results.to_csv(
        output_path / "spearman_sensitivity_results.csv", index=True
    )
    spearman_results.to_json(
        output_path / "spearman_sensitivity_results.json", orient="index", indent=2
    )

    # Save Sobol results (dict of DataFrames)
    for output_name, df in sobol_results.items():
        df.to_csv(output_path / f"sobol_sensitivity_{output_name}.csv", index=True)

    with open(output_path / "sobol_sensitivity_results.json", "w") as f:
        json.dump(
            {k: v.to_dict(orient="index") for k, v in sobol_results.items()},
            f,
            indent=2,
        )

    # Save Morris results (dict of DataFrames)
    for output_name, df in morris_results.items():
        df.to_csv(output_path / f"morris_sensitivity_{output_name}.csv", index=True)

    with open(output_path / "morris_sensitivity_results.json", "w") as f:
        json.dump(
            {k: v.to_dict(orient="index") for k, v in morris_results.items()},
            f,
            indent=2,
        )

    # Save RF Permutation results (nested dict with Series)
    for output_name, metrics in rf_results.items():
        for metric_name, series in metrics.items():
            series.to_csv(
                output_path / f"rf_permutation_{output_name}_{metric_name}.csv",
                header=True,
            )

    rf_data = {
        output: {k: v.to_dict() for k, v in metrics.items()}
        for output, metrics in rf_results.items()
    }
    with open(output_path / "rf_permutation_sensitivity_results.json", "w") as f:
        json.dump(rf_data, f, indent=2)

    logger.info("All sensitivity results saved successfully")
