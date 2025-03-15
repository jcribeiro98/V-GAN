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


def pipeline_time(features: list, seeds=[777, 1234, 12345], base_subspace_selector: ss.BaseSubspaceSelector = ss.HiCS()):
    """Pipeline for the time experiments

    This function will run the time experiments in a collection of datasets and for a group of outlier detection models.
    Additionally, the funcition is capable of, given the existence of a version of VGAN, load it and use it for its own porpuse

    Args:
        datasets (list): list of datasets to run the experiments in
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    features = np.linspace(100, 10000, 10, dtype=int)

    # To get CAE's result use a diff. envieronment with keras
    for base_method in [VGAN(epochs=2000, device='cpu'), ss.CLIQUE(), ss.GMD(), ss.HiCS()]:
        pipeline_time(
            features, base_subspace_selector=base_method)
