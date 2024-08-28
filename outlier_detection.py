from src.modules.od_module import VGAN
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.feature_bagging import FeatureBagging
from pathlib import Path
import datetime
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score, f1_score
from sel_suod.models.base import sel_SUOD
import itertools
from sklearn.preprocessing import label_binarize
from joblib.externals.loky import get_reusable_executor
from data.get_datasets import load_data
import json
import os
import logging
logger = logging.getLogger(__name__)


def launch_outlier_detection_experiments(dataset_name: str, base_estimators: list, epochs=10, temperature=10) -> dict:
    """Launch the outlier detection experiments for a given dataset

    Args:
        dataset_name (str): Name of the dataset to load
        base_estimators (list): List including all base estimators to build the ensemble. If the length is = 1, then an homogeneus ensemble will be fitted.
        directory (Path): Path to the directory one wishes to load to
    Returns:
        tuple: Returns the AUC, PRAUC and F1 of the ensemble obtained by VGAN subspaces
    """
    logger.info(
        "No instance of a pretrained generation model found. Proceeding to train a new Generator.")
    X_train, X_test, y_test = load_data(dataset_name)

    vgan = VGAN(epochs=epochs, temperature=temperature, batch_size=500,
                path_to_directory=Path() / "experiments" /
                f"VGAN_{dataset_name}",
                iternum_d=1, iternum_g=5, lr_G=0.01, lr_D=0.01)
    vgan.fit(X_train)
    vgan.approx_subspace_dist()
    ensemble_model = sel_SUOD(base_estimators=base_estimators, subspaces=vgan.subspaces,
                              n_jobs=5, bps_flag=False, approx_flag_global=False)
    ensemble_model.fit(X_train)
    decision_function_scores_ens = ensemble_model.decision_function(
        X_test)
    decision_function_scores_ens = np.average(
        decision_function_scores_ens, axis=1, weights=vgan.proba)

    return {"Dataset": dataset_name,
            "AUC": auc(y_test, decision_function_scores_ens),
            "PRAUC": average_precision_score(y_test, decision_function_scores_ens),
            "F1": f1_score(y_test, (decision_function_scores_ens > np.quantile(decision_function_scores_ens, .80)) * 1)}


def pretrained_launch_outlier_detection_experiments(dataset_name: str, base_estimators: list) -> tuple:
    """Launch the outlier detection experiments for a given dataset

    Args:
        dataset_name (str): Name of the dataset to load
        base_estimators (list): List including all base estimators to build the ensemble. If the length is = 1, then an homogeneus ensemble will be fitted.
        directory (Path): Path to the directory one wishes to load to
    Returns:
        np.array: Returns the AUC, PRAUC and F1
    """
    logger.info(
        f"Pretrained generator found!")
    X_train, X_test, y_test = load_data(dataset_name)

    # TO DO: Add a function to launch outlier detection with a pretrained instance of VGAN/VMMD (to save training time) using the load_models option.
    vgan = VGAN()
    vgan.load_models(Path() / "experiments" /
                     f"VGAN_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    vgan.approx_subspace_dist()
    ensemble_model = sel_SUOD(base_estimators=base_estimators, subspaces=vgan.subspaces,
                              n_jobs=5, bps_flag=False, approx_flag_global=False)
    ensemble_model.fit(X_train)
    decision_function_scores_ens = ensemble_model.decision_function(
        X_test)
    decision_function_scores_ens = np.average(
        decision_function_scores_ens, axis=1, weights=vgan.proba)

    return {"Dataset": dataset_name,
            "AUC": auc(y_test, decision_function_scores_ens),
            "PRAUC": average_precision_score(y_test, decision_function_scores_ens),
            "F1": f1_score(y_test, (decision_function_scores_ens > np.quantile(decision_function_scores_ens, .80)) * 1)}


# TO DO: Add a function to launch outlier detection with a pretrained instance of VGAN/VMMD (to save training time)


if __name__ == "__main__":
    dataset_name = "speech"

    auc_vgan_ens = launch_outlier_detection_experiments(dataset_name, [
        LOF()])
    print(
        f'AUC obtained by the VGAN-based ensemble model: {print(auc_vgan_ens)}')

    model = LOF()
    X_train, X_test, y_test = load_data(dataset_name)
    model.fit(X_train)
    decision_function_scores = model.decision_function(X_test)
    auc_model = auc(y_test, decision_function_scores)
    print(f'AUC obtained by unsembled model: {auc_model}')
