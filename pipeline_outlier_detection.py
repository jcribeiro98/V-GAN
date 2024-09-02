from src.modules.od_module import VGAN
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
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
import random
from sklearn.metrics import average_precision_score, f1_score
import os
from outlier_detection import launch_outlier_detection_experiments, pretrained_launch_outlier_detection_experiments, check_if_myopicity_was_uphold
import logging
from colorama import Fore
logger = logging.getLogger(__name__)


def pipeline_outlier_detection_vgan(datasets: list, outlier_detection_models: list = None, experimental_settings: np.array = None,
                                    base_methods: list = [], seeds=[777, 1234, 12345, 000, 1000, 20, 30, 33, 90, 10], gen_model_to_use: str = "VGAN"):
    """Pipeline for the outlier detection experiments

    This function will run the outlier detection experiments in a collection of datasets and for a group of outlier detection models.
    Additionally, the funcition is capable of, given the existence of a version of VGAN/VMMD, load it and use it for its own porpuse

    Args:
        datasets (list): list of datasets to run the experiments in
        outlier_detection_models (list): list of outlier detection models to run the experiments withs (besides vgan ensemble)
        base_methods (list): List of base_estimators to employ as the basis of the ensemble used in VGAN. It does NOT function as in SUOD and sel_SUOD,
        but rather as FeatureBagging, where every model introduced here will be used in SEPARATE homogeneous ensembles.
    """
    results_df = pd.DataFrame(
        {"Dataset": [], "Method": [], "AUC": [], "PRAUC": [], "F1": []})
    for i, dataset in enumerate(datasets):
        for j, base_method in enumerate(base_methods):
            logger.info(
                f"Running dataset {Fore.CYAN}{dataset}{Fore.RESET}, number {i+1} out of {datasets.__len__()} using method: {base_method.__class__.__name__} with VGAN, number {j+1} out of {len(base_methods)}")
            for k, seed in enumerate(seeds):
                logger.info(f"Seed number {Fore.CYAN}{k+1}{Fore.RESET}")
                try:

                    if os.path.isdir(Path() / "experiments" / f"{gen_model_to_use}" / f"{gen_model_to_use}_{dataset}"):
                        logger.info(
                            f"Launching {gen_model_to_use} experiments...")
                        results_dict = pretrained_launch_outlier_detection_experiments(
                            dataset_name=dataset, base_estimators=[base_method], seed=seed, gen_model_to_use=gen_model_to_use)
                    else:
                        results_dict = launch_outlier_detection_experiments(
                            dataset_name=dataset, base_estimators=[
                                base_method],
                            epochs=2000, temperature=10, seed=seed, gen_model_to_use=gen_model_to_use)
                    results_dict["Method"] = base_method.__class__.__name__
                    results_df = results_df._append(
                        results_dict, ignore_index=True)
                except:
                    logger.warning(
                        f"Error found during exection in dataset: {dataset}")
                pass

                results_df.to_csv(Path() / "experiments" /
                                  "Outlier_Detection" / f"Results_{gen_model_to_use}_{[method.__class__.__name__ for method in base_methods].__repr__()}_all_datasets.csv")


def pipeline_outlier_detection_classic_od(datasets: list, outlier_detection_models: list = None, experimental_settings:
                                          np.array = None, base_methods: list = [], seeds=[777, 1234, 12345, 11, 1000, 20, 30, 33, 90, 10]):
    """Pipeline for the outlier detection experiments

    This function will run the outlier detection experiments in a collection of datasets and for a group of outlier detection models.
    Additionally, the funcition is capable of, given the existence of a version of VGAN/VMMD, load it and use it for its own porpuse

    Args:
        datasets (list): list of datasets to run the experiments in
        outlier_detection_models (list): list of outlier detection models to run the experiments withs (besides vgan ensemble)
        base_methods (list): List of base_estimators to employ as the basis of the ensemble used in VGAN. It does NOT function as in SUOD and sel_SUOD,
        but rather as FeatureBagging, where every model introduced here will be used in SEPARATE homogeneous ensembles.
    """
    results_df = pd.DataFrame(
        {"Dataset": [], "Method": [], "AUC": [], "PRAUC": [], "F1": []})
    for i, dataset in enumerate(datasets):
        for j, base_method in enumerate(base_methods):
            logger.info(
                f"Running dataset {Fore.CYAN}{dataset}{Fore.RESET}, number {i+1} out of {datasets.__len__()} using method: {base_method.__class__.__name__} unensembled, number {j+1} out of {len(base_methods)}")
            for seed in seeds:
                try:

                    X_train, X_test, y_test = load_data(dataset)

                    base_method.fit(X_train)
                    decision_function_scores = base_method.decision_function(
                        X_test)

                    results_dict = {"Dataset": dataset,
                                    "AUC": auc(y_test, decision_function_scores),
                                    "PRAUC": average_precision_score(y_test, decision_function_scores),
                                    "F1": f1_score(y_test, (decision_function_scores > np.quantile(decision_function_scores, .80)) * 1)}

                    results_dict["Method"] = base_method.__class__.__name__

                    results_df = results_df._append(
                        results_dict, ignore_index=True)
                except:
                    logger.warning(
                        f"Error found during exection in dataset: {dataset} with method {base_method.__class__.__name__}")
                    pass

        results_df.to_csv(Path() / "experiments" /
                          "Outlier_Detection" / f"Results_{[method.__class__.__name__ for method in base_methods].__repr__()}_all_datasets.csv")


def pipeline_outlier_detection_ens_od(datasets: list, outlier_detection_models: list = None, experimental_settings: np.array = None,
                                      base_methods: list = [], seeds=[777, 1234, 12345, 000, 1000, 20, 30, 33, 90, 10]):
    """Pipeline for the outlier detection experiments

    This function will run the outlier detection experiments in a collection of datasets and for a group of outlier detection models.
    Additionally, the funcition is capable of, given the existence of a version of VGAN/VMMD, load it and use it for its own porpuse

    Args:
        datasets (list): list of datasets to run the experiments in
        outlier_detection_models (list): list of outlier detection models to run the experiments withs (besides vgan ensemble)
        base_methods (list): List of base_estimators to employ as the basis of the ensemble used in VGAN. It does NOT function as in SUOD and sel_SUOD,
        but rather as FeatureBagging, where every model introduced here will be used in SEPARATE homogeneous ensembles.
    """
    results_df = pd.DataFrame(
        {"Dataset": [], "Method": [], "AUC": [], "PRAUC": [], "F1": []})
    for i, dataset in enumerate(datasets):
        for j, base_method in enumerate(base_methods):
            logger.info(
                f"Running dataset {Fore.CYAN}{dataset}{Fore.RESET}, number {i+1} out of {datasets.__len__()} using method: {base_method.__class__.__name__} with Feature Bagging, number {j+1} out of {len(base_methods)}")
            for seed in seeds:
                try:
                    X_train, X_test, y_test = load_data(dataset)
                    ensemble_method = FeatureBagging(
                        base_estimator=base_method, n_estimators=100, random_state=seed)
                    ensemble_method.fit(X_train)
                    decision_function_scores = ensemble_method.decision_function(
                        X_test)

                    results_dict = {"Dataset": dataset,
                                    "AUC": auc(y_test, decision_function_scores),
                                    "PRAUC": average_precision_score(y_test, decision_function_scores),
                                    "F1": f1_score(y_test, (decision_function_scores > np.quantile(decision_function_scores, .80)) * 1)}

                    results_dict["Method"] = base_method.__class__.__name__

                    results_df = results_df._append(
                        results_dict, ignore_index=True)
                except:
                    logger.warning(
                        f"Error found during exection in dataset: {dataset} with method {base_method.__class__.__name__}")
                    pass

        results_df.to_csv(Path() / "experiments" /
                          "Outlier_Detection" / f"Results_ENSEMBLED_{[method.__class__.__name__ for method in base_methods].__repr__()}_all_datasets.csv")


def pipeline_gof_test(datasets, gen_model_to_use="VGAN"):

    results_df = pd.DataFrame(
        {"Dataset": [], "p-value": [], "subspace_count": []})
    for i, dataset in enumerate(datasets):
        logger.info(
            f"Runnig GOF test for {gen_model_to_use} in dataset: {Fore.CYAN}{dataset}{Fore.RESET}; number {i+1} out of {datasets.__len__()}")

        try:
            pvalue, count = check_if_myopicity_was_uphold(
                dataset_name=dataset, gen_model_to_use=gen_model_to_use)
            results_dict = {"Dataset": dataset,
                            "p-value": pvalue,
                            "subspace_count": count}
            results_df = results_df._append(
                results_dict, ignore_index=True)
            results_df.to_csv(Path() /
                              "experiments" / "Outlier_Detection" / f"Results_GOF_{gen_model_to_use}_all_datasets.csv")
        except:
            logger.warning(
                f"Error found during exection in dataset: {dataset}")
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    datasets = ["20news_0",
                "MVTec-AD_bottle",
                "InternetAds",
                "agnews_0",
                "amazon",
                "imdb",
                "yelp",
                "MNIST-C_brightness",
                "CIFAR10_0",
                "FashionMNIST_0",
                "SVHN_0",
                "speech",
                "musk",
                "mnist",
                "optdigits",
                "SpamBase",
                "landsat",
                "satellite",
                "satimage-2",
                "Ionosphere",
                "WPBC",
                "letter",
                "WDBC",
                "fault",
                "annthyroid",
                "cardio",
                "Cardiotocography",
                "Waveform",
                "Hepatitis",
                "Lymphography",
                "pendigits",
                "wine",
                "vowels",
                "PageBlocks",
                "breastw",
                "Stamps",
                "WBC",
                "Pima",
                "yeast",
                "thyroid",
                "vertebral",
                "Wilt"]

    pipeline_outlier_detection_vgan(
        datasets, base_methods=[CBLOF()], gen_model_to_use="VGAN")
    pipeline_outlier_detection_vgan(
        datasets, base_methods=[ECOD()], gen_model_to_use="VGAN")
    # pipeline_outlier_detection_vgan(datasets, base_methods=[KNN()], gen_model_to_use="VMMD")
    # pipeline_gof_test(datasets=datasets, gen_model_to_use="VMMD")
    # pipeline_gof_test(datasets=datasets, gen_model_to_use="VGAN")
    # pipeline_outlier_detection_classic_od(datasets, base_methods={LOF(), KNN(), ECOD()})
    pipeline_outlier_detection_ens_od(datasets, base_methods={CBLOF()})
    pipeline_outlier_detection_ens_od(datasets, base_methods={ECOD()})
