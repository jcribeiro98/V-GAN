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
from sel_suod.models.base import sel_SUOD
import itertools
from sklearn.preprocessing import label_binarize
from joblib.externals.loky import get_reusable_executor
from data.get_datasets import load_data
import json
import os


def lauch_outlier_detection_experiments(dataset_name: str, base_estimators: list, epochs=2000, temperature=10) -> np.array:
    """Launch the outlier detection experiments for a given dataset

    Args:
        dataset_name (str): Name of the dataset to load
        base_estimators (list): List including all base estimators to build the ensemble. If the length is = 1, then an homogeneus ensemble will be fitted.
        directory (Path): Path to the directory one wishes to load to
    Returns:
        np.array: Returns the AUC, PRAUC and F1
    """
    X_train, X_test, y_test = load_data(dataset_name)

    vgan = VGAN(epochs=epochs, temperature=temperature, batch_size=500,
                path_to_directory=Path() / "experiments" /
                f"Out_Detection_{dataset_name}_{datetime.datetime.now()}",
                iternum_d=1, iternum_g=5, lr_G=0.01, lr_D=0.01)
    vgan.fit(X_train)
    vgan.approx_subspace_dist()
    print(pd.DataFrame(vgan.subspaces))
    ensemble_model = sel_SUOD(base_estimators=base_estimators, subspaces=vgan.subspaces,
                              n_jobs=5, bps_flag=False, approx_flag_global=False)
    ensemble_model.fit(X_train)
    decision_function_scores_ens = ensemble_model.decision_function(
        X_test)
    decision_function_scores_ens = np.average(
        decision_function_scores_ens, axis=1, weights=vgan.proba)

    return vgan, auc(y_test, decision_function_scores_ens)


if __name__ == "__main__":
    # X_train = pd.read_parquet(
    #     "data/train_p53_mutant.parquet").to_numpy()[:, range(5)]
    # X_train = normalize(X_train, axis=0)

    # vgan = VGAN(epochs=1, temperature=1, batch_size=500,
    #             path_to_directory=Path() / "experiments" /
    #             f"Out_Detection_p53_{datetime.datetime.now()}",
    #             iternum_d=1, iternum_g=5, lr_G=0.01, lr_D=0.01)
    # vgan.fit(X_train)

    # vgan.approx_subspace_dist()
    # model = sel_SUOD(base_estimators=[LOF()], subspaces=vgan.subspaces,
    #                  n_jobs=5, bps_flag=False, approx_flag_global=False)
    # model.fit(X_train)
    # X_test = pd.read_parquet(
    #     "data/test_p53_mutant.parquet").to_numpy()[:, range(5)]
    # y_test = pd.read_csv("data/test_p53_mutant_gt.csv").to_numpy()
    # y_test = label_binarize(
    #     y_test[:, 1], classes=["inactive", "active"])
    # decision_function_scores_ens = model.decision_function(X_test)
    # get_reusable_executor().shutdown(wait=True)
    # decision_function_scores_ens = np.average(
    #     decision_function_scores_ens, axis=1, weights=vgan.proba)
    dataset_name = "speech"

    vgan, auc_vgan_ens = lauch_outlier_detection_experiments(dataset_name, [
                                                             LOF()])
    print(f'AUC obtained by the VGAN-based ensemble model: {auc_vgan_ens}')

    model = LOF()
    X_train, X_test, y_test = load_data(dataset_name)
    model.fit(X_train)
    decision_function_scores = model.decision_function(X_test)
    auc_model = auc(y_test, decision_function_scores)
    print(f'AUC obtained by unsembled model: {auc_model}')

    model = LOF()
    fb_model = FeatureBagging(model, vgan.subspaces.shape[0], n_jobs=5)
    fb_model.fit(X_train)
    decision_function_scores_fb = fb_model.decision_function(X_test)
    auc_fb_model = auc(y_test, decision_function_scores_fb)
    print(f'AUC obtained by unsembled model: {auc_fb_model}')
