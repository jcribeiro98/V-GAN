from src.modules.synthetic_module import generate_data
import signal
import time
import src.modules.ss_module as ss
from colorama import Fore
import logging
from sys import exit
from src.modules.od_module import VGAN
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from data.get_datasets import load_data
import random
from sklearn.metrics import average_precision_score, f1_score
import os
import torch
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1
os.environ["CUDA_VISIBLE_DEVICES"] = ","
logger = logging.getLogger(__name__)


def timeout_call(signum, frame):
    raise TimeoutError("Time excedeed the intended execution time")


def pipeline_time(features: list, outlier_detection_models: list = None, experimental_settings: np.array = None,
                  base_methods: list = [], seeds=[777, 1234, 12345], base_subspace_selector: ss.BaseSubspaceSelector = ss.HiCS()):
    """Pipeline for the outlier detection experiments

    This function will run the outlier detection experiments in a collection of datasets and for a group of outlier detection models.
    Additionally, the funcition is capable of, given the existence of a version of VGAN/VMMD, load it and use it for its own porpuse

    Args:
        datasets (list): list of datasets to run the experiments in
        outlier_detection_models (list): list of outlier detection models to run the experiments withs (besides vgan ensemble)
        base_methods (list): List of base_estimators to employ as the basis of the ensemble used in VGAN. It does NOT function as in SUOD and sel_SUOD,
        but rather as FeatureBagging, where every model introduced here will be used in SEPARATE homogeneous ensembles.
    """
    torch.set_num_threads(1)
    time_df = pd.DataFrame(
        {"Dataset": [], "Time": []}
    )
    for i, feature_count in enumerate(features):
        try:
            for j, seed in enumerate(seeds):
                logger.info(
                    f"Running synth. dataset wtih {Fore.CYAN}{feature_count}{Fore.RESET} features, using method: {Fore.CYAN}{base_subspace_selector.__class__.__name__}{Fore.RESET}")
                X_train = generate_data(feature_count=feature_count)
                subspace_selector = base_subspace_selector
                path_times = Path() / "experiments" / "Synthetic" / "Times" / \
                    f"Times_{
                        subspace_selector.__class__.__name__}_all_datasets.csv"

                signal.signal(signal.SIGALRM, handler=timeout_call)
                tic = time.time()
                signal.alarm(36000)
                subspace_selector.fit(X_train)
                toc = time.time()
                signal.alarm(0)
                time_dict = {"Features": feature_count, "Time": toc-tic}
                time_df = time_df._append(time_dict, ignore_index=True)
                time_df.to_csv(path_times)

        except TimeoutError:
            signal.alarm(0)
            toc = time.time()
            logger.warning(
                f"Timeout for ss method {subspace_selector.__class__.__name__} with {feature_count} features!")
            time_dict = {"Features": feature_count, "Time": 36000}
            time_df = time_df._append(time_dict, ignore_index=True)
            time_df.to_csv(path_times)

            break


def pipeline_outlier_detection_dimred_od(datasets: list, base_subspace_selector: ss.BaseSubspaceSelector,  outlier_detection_models: list = None, experimental_settings: np.array = None,
                                         base_methods: list = [], seeds=[777, 1234, 12345, 000, 1000, 20, 30, 33, 90, 10]):

    results_df = pd.DataFrame(
        {"Dataset": [], "Method": [], "AUC": [], "PRAUC": [], "F1": []})
    time_df = pd.DataFrame(
        {"Dataset": [], "Time": []}
    )
    path_results = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / \
        f"Results_{base_subspace_selector.__class__.__name__}_{
            [method.__class__.__name__ for method in base_methods].__repr__()}_all_datasets.csv"
    path_times = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / \
        f"Times_{base_subspace_selector.__class__.__name__}_all_datasets.csv"

    for i, dataset in enumerate(datasets):
        for j, base_method in enumerate(base_methods):
            logger.info(
                f"Running dataset {Fore.CYAN}{dataset}{Fore.RESET}, number {i+1} out of {datasets.__len__()} using method: {base_method.__class__.__name__} with {Fore.CYAN}{base_subspace_selector.__class__.__name__}{Fore.RESET}, number {j+1} out of {len(base_methods)}")
            for seed in seeds:

                X_train, X_test, y_test = load_data(dataset)
                subspace_selector = base_subspace_selector
                subspace_selector.random_state = seed
                tic = time.time()
                subspace_selector.fit(X_train)
                toc = time.time()

                time_dict = {"Dataset": dataset, "Time": toc-tic}
                time_df = time_df._append(time_dict, ignore_index=True)
                time_df.to_csv(path_times)

                try:
                    random.seed(seed)
                    subspace_selector.fit_odm(
                        X_train=X_train, base_odm=base_method)
                    decision_function_scores = subspace_selector.decision_function_odm(
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
                    continue

                results_df.to_csv(path_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    features = np.linspace(100, 10000, 10, dtype=int)

    # To get CAE's result use a diff. envieronment with keras
    for base_method in [VGAN(epochs=2000, device='cpu'), ss.CLIQUE(), ss.GMD()]:
        pipeline_time(
            features, base_subspace_selector=base_method)
