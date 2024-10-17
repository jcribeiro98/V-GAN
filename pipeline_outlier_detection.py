from src.modules.od_module import VGAN
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from pathlib import Path
import datetime
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sel_suod.models.base import sel_SUOD
import itertools
from src.modules.tools import numeric_to_boolean, aggregator_funct
from sklearn.preprocessing import label_binarize
from joblib.externals.loky import get_reusable_executor
from data.get_datasets import load_data
import json
from pyod.utils.utility import generate_bagging_indices
import random
from sklearn.metrics import average_precision_score, f1_score
import os
from sys import exit
from outlier_detection import launch_outlier_detection_experiments, pretrained_launch_outlier_detection_experiments, check_if_myopicity_was_uphold
import logging
from colorama import Fore
from sklearn.base import clone
import src.modules.ss_module as ss
import time
import signal
from src.modules.synthetic_module import generate_data

logger = logging.getLogger(__name__)


def timeout_call(signum, frame):
    raise TimeoutError("Time excedeed the intended execution time")


def pipeline_time(features: list, outlier_detection_models: list = None, experimental_settings: np.array = None,
                                      base_methods: list = [], seeds=[777, 1234, 12345, 000, 1000], base_subspace_selector: ss.BaseSubspaceSelector = ss.HiCS()):
    """Pipeline for the outlier detection experiments

    This function will run the outlier detection experiments in a collection of datasets and for a group of outlier detection models.
    Additionally, the funcition is capable of, given the existence of a version of VGAN/VMMD, load it and use it for its own porpuse

    Args:
        datasets (list): list of datasets to run the experiments in
        outlier_detection_models (list): list of outlier detection models to run the experiments withs (besides vgan ensemble)
        base_methods (list): List of base_estimators to employ as the basis of the ensemble used in VGAN. It does NOT function as in SUOD and sel_SUOD,
        but rather as FeatureBagging, where every model introduced here will be used in SEPARATE homogeneous ensembles.
    """
    time_df = pd.DataFrame(
        {"Dataset": [], "Time": []}
    )
    for i, feature_count in enumerate(features):
        for j, seed in enumerate(seeds):
            logger.info(
                f"Running synth. dataset wtih {Fore.CYAN}{feature_count}{Fore.RESET} features, using method: {Fore.CYAN}{base_subspace_selector.__class__.__name__}{Fore.RESET}")
            X_train = generate_data(feature_count=feature_count)
            subspace_selector = base_subspace_selector
            path_times = Path() / "experiments" / "Synthetic" / "Times"/ \
                f"Times_{subspace_selector.__class__.__name__}_all_datasets.csv"
            try:
                    signal.signal(signal.SIGALRM, handler=timeout_call)
                    tic = time.time()
                    signal.alarm(18000)
                    subspace_selector.fit(X_train)
                    toc = time.time()
                    signal.alarm(0)

            except TimeoutError:
                    signal.alarm(0)
                    toc = time.time()
                    logger.warning(
                        f"Timeout for ss method {subspace_selector.__class__.__name__} with {feature_count} features!")
                    time_dict = {"Features": feature_count, "Time": 18000}
                    time_df = time_df._append(time_dict, ignore_index=True)
                    time_df.to_csv(path_times)
                    exit("Takes too much time! ")
            time_dict = {"Features": feature_count, "Time": toc-tic}
            time_df = time_df._append(time_dict, ignore_index=True)
            time_df.to_csv(path_times)

    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    features = [ 100,  325,  550,  775, 1000, 3250, 5500, 7750, 10000]

    for base_method in [ss.HiCS(),ss.CLIQUE(),VGAN(epochs=2000)]:
        pipeline_time(
            features, base_subspace_selector=base_method)