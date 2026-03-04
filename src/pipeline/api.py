import logging
import os

from joblib import Parallel, delayed

from utils import save_sensitivity_results, setup_logging

logger = logging.getLogger(__name__)

SENSITIVITY_METHODS = {}


def register_sensitivity_method(name=None):
    def decorator(func):
        method_name = name or func.__name__

        if method_name in SENSITIVITY_METHODS:
            raise ValueError(f"Sensitivity method '{method_name}' already registered")

        SENSITIVITY_METHODS[method_name] = func
        logger.info("Registered sensitivity method: %s", method_name)

        return func

    return decorator


def run_method(name, func, params, results):
    os.environ["HOCLOOP_LOG_SUFFIX"] = f"_{name}"
    setup_logging()

    logger.info("Running %s sensitivity analysis", name)

    result = func(params, results)

    save_sensitivity_results(**{name: result})

    logger.info("%s finished", name)

    return name


def run_sensitivity_pipeline(params, results, n_jobs=-1):
    Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_method)(name, func, params, results)
        for name, func in SENSITIVITY_METHODS.items()
    )
