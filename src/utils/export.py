import logging
import json
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

logger = logging.getLogger(__name__)


def _save_dataframe(df: pd.DataFrame, base_name: str, output_path: Path) -> None:
    """Save a DataFrame to CSV and JSON."""
    df.to_csv(output_path / f"{base_name}.csv", index=True)
    df.to_json(output_path / f"{base_name}.json", orient="index", indent=2)


def _save_dict_of_dataframes(
    data: Dict[str, pd.DataFrame], base_name: str, output_path: Path
) -> None:
    """Save a dictionary of DataFrames to individual CSVs and combined JSON."""
    # Save individual CSV files per output
    for output_name, df in data.items():
        df.to_csv(output_path / f"{base_name}_{output_name}.csv", index=True)

    # Save combined JSON
    combined_json = {k: v.to_dict(orient="index") for k, v in data.items()}
    with open(output_path / f"{base_name}_results.json", "w") as f:
        json.dump(combined_json, f, indent=2)


def _save_nested_dict(
    data: Dict[str, Dict[str, pd.Series]], base_name: str, output_path: Path
) -> None:
    """Save nested dictionary with Series to individual CSVs and combined JSON."""
    # Save individual CSV files per output and metric
    for output_name, metrics in data.items():
        for metric_name, series in metrics.items():
            series.to_csv(
                output_path / f"{base_name}_{output_name}_{metric_name}.csv",
                header=True,
            )

    # Save combined JSON
    combined_json = {
        output: {k: v.to_dict() for k, v in metrics.items()}
        for output, metrics in data.items()
    }
    with open(output_path / f"{base_name}_results.json", "w") as f:
        json.dump(combined_json, f, indent=2)


def _dataframe_to_dict(df: pd.DataFrame) -> dict:
    """Convert DataFrame to dict, handling different index types."""
    if isinstance(df.index[0], tuple):
        # For multi-index or tuple indices, convert tuples to strings
        return {str(k): v for k, v in df.to_dict(orient="index").items()}
    return df.to_dict(orient="index")


def _save_pce_structure(
    data: Dict[str, Dict[str, Any]], base_name: str, output_path: Path
) -> None:
    """
    Save PCE-specific nested structure with main_effects and interactions.

    Structure: {output: {"main_effects": DataFrame, "interactions": {order: DataFrame}}}
    """
    combined_json = {}

    for output_name, components in data.items():
        # Save main effects
        main_effects = components.get("main_effects")
        if isinstance(main_effects, pd.DataFrame):
            main_effects.to_csv(
                output_path / f"{base_name}_{output_name}_main_effects.csv", index=True
            )
            combined_json.setdefault(output_name, {})["main_effects"] = (
                _dataframe_to_dict(main_effects)
            )

        # Save interactions
        interactions = components.get("interactions", {})
        if interactions:
            combined_json.setdefault(output_name, {})["interactions"] = {}

            for order_name, interaction_df in interactions.items():
                if isinstance(interaction_df, pd.DataFrame):
                    interaction_df.to_csv(
                        output_path / f"{base_name}_{output_name}_{order_name}.csv",
                        index=False,  # Interactions often have tuple indices
                    )
                    combined_json[output_name]["interactions"][order_name] = (
                        interaction_df.to_dict(orient="records")
                    )

    # Save combined JSON
    with open(output_path / f"{base_name}_results.json", "w") as f:
        json.dump(combined_json, f, indent=2)


def save_sensitivity_results(
    output_dir: str = "results",
    **results: Union[pd.DataFrame, Dict[str, Any]],
) -> None:
    """
    Save sensitivity analysis results to CSV and JSON files.

    This function handles different result types automatically:
    - DataFrame: saved as single CSV + JSON
    - Dict[str, DataFrame]: saved as multiple CSVs (one per key) + combined JSON
    - Dict[str, Dict[str, Series]]: saved as multiple CSVs + combined JSON
    - Dict[str, Dict[str, Any]]: PCE-style nested structure with main_effects and interactions

    Parameters
    ----------
    output_dir : str
        Directory to save results to
    **results : Union[pd.DataFrame, Dict[str, Any]]
        Keyword arguments where keys are method names and values are results

    Examples
    --------
    >>> save_sensitivity_results(
    ...     spearman=spearman_results,
    ...     sobol=sobol_results,
    ...     morris=morris_results,
    ...     rf_permutation=rf_results,
    ...     pce=pce_results,
    ...     hdmr=hdmr_results,
    ... )
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    logger.info("Saving sensitivity results to %s", output_dir)

    for method_name, result in results.items():
        if result is None:
            logger.warning("Skipping %s: result is None", method_name)
            continue

        try:
            if isinstance(result, pd.DataFrame):
                # Simple DataFrame
                _save_dataframe(
                    result, f"{method_name}_sensitivity_results", output_path
                )
                logger.debug("Saved %s results (DataFrame)", method_name)

            elif isinstance(result, dict):
                # Check if it's a dict of DataFrames or nested dict
                first_value = next(iter(result.values()))

                if isinstance(first_value, pd.DataFrame):
                    # Dict[str, DataFrame] - e.g., Sobol, Morris, HDMR
                    _save_dict_of_dataframes(
                        result, f"{method_name}_sensitivity", output_path
                    )
                    logger.debug("Saved %s results (dict of DataFrames)", method_name)

                elif isinstance(first_value, dict):
                    # Check if it's PCE-style structure or RF-style
                    inner_keys = set(first_value.keys())

                    if "main_effects" in inner_keys or "interactions" in inner_keys:
                        # PCE-style: {output: {"main_effects": df, "interactions": {...}}}
                        _save_pce_structure(
                            result, f"{method_name}_sensitivity", output_path
                        )
                        logger.debug("Saved %s results (PCE structure)", method_name)

                    else:
                        # RF-style: Dict[str, Dict[str, Series]]
                        inner_value = next(iter(first_value.values()))
                        if isinstance(inner_value, (pd.Series, pd.DataFrame)):
                            _save_nested_dict(
                                result, f"{method_name}_sensitivity", output_path
                            )
                            logger.debug("Saved %s results (nested dict)", method_name)
                        else:
                            logger.warning(
                                "Skipping %s: unsupported nested dict structure",
                                method_name,
                            )

                else:
                    logger.warning(
                        "Skipping %s: unsupported dict value type %s",
                        method_name,
                        type(first_value).__name__,
                    )
            else:
                logger.warning(
                    "Skipping %s: unsupported result type %s",
                    method_name,
                    type(result).__name__,
                )

        except Exception as e:
            logger.error("Error saving %s results: %s", method_name, e, exc_info=True)

    logger.info("All sensitivity results saved successfully")
