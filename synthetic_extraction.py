from src.modules.od_module import VGAN, VMMD
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
from src.modules.synthetic_module import generate_data_3d
import logging
from src.modules.tools import aggregator_funct
logger = logging.getLogger(__name__)


def launch_outlier_detection_experiments(freq: int, base_estimators: list, epochs: int = 3000,
                                         temperature: float = 0, seed: int = 777, gen_model_to_use: str = "VGAN") -> dict:
    """Launch the outlier detection experiments for a given dataset

    Args:
        dataset_name (str): Name of the dataset to load
        base_estimators (list): List including all base estimators to build the ensemble. If the length is = 1, 
        then an homogeneus ensemble will be fitted.
        directory (Path): Path to the directory one wishes to load to
    Returns:
        tuple: Returns the AUC, PRAUC and F1 of the ensemble obtained by VGAN subspaces
    """

    X_train = generate_data_3d(freq, 10000, seed=seed)
    # X_train, X_test, y_test = load_data("Ionosphere")

    if gen_model_to_use == "VGAN":
        vgan = VGAN(epochs=epochs, temperature=temperature, batch_size=1000,
                    path_to_directory=Path() / "experiments" / "Synthetic" / "VGAN" /
                    f"VGAN_{freq}",
                    iternum_d=1, iternum_g=5, lr_G=0.01, lr_D=0.01)
    elif gen_model_to_use == "VMMD":
        vgan = VMMD(epochs=epochs,  batch_size=6000,
                    path_to_directory=Path() / "experiments" / "Synthetic" / "VMMD" /
                    f"VMMD_{freq}", lr=0.01)
    else:
        raise ValueError(f"{gen_model_to_use} is not a generator in the list")

    vgan.fit(X_train)
    vgan.seed = seed
    vgan.approx_subspace_dist(add_leftover_features=False)
    print(pd.DataFrame(vgan.subspaces, vgan.proba))
    proba_p2 = vgan.proba[[vgan.subspaces[i].tolist() in [[True, False, True], [
                          False, False, True], [False, True, True]] for i in range(vgan.subspaces.__len__())]].sum()
    proba_p1 = vgan.proba[[vgan.subspaces[i].tolist(
    ) in [[True, True, False]] for i in range(vgan.subspaces.__len__())]].sum()

    return proba_p1, proba_p2


def check_if_myopicity_was_uphold(dataset_name: str, gen_model_to_use="VGAN") -> tuple:
    """Given the dataset name, the function will return the p-value of the GOF test using the MMD with the recommended 
    bandwidth in [Look for the paper I don't remember the surname of the authors rn].

    Args:
        dataset_name (str): Name of the dataset as included in the json file 'datasets_file_name.json'

    Returns:
        float: p-value for the two-sampe non-parametric GoF test using the MMD with recommended bandwidth (by L2 distances)
    """
    X_train, _, _ = load_data(dataset_name)

    if gen_model_to_use == "VGAN":
        vgan = VGAN()
        vgan.load_models(Path() / "experiments" / "VGAN" /
                         f"VGAN_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    elif gen_model_to_use == "VMMD":
        vgan = VMMD()
        vgan.load_models(Path() / "experiments" / "VMMD" /
                         f"VMMD_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    vgan.approx_subspace_dist()

    return vgan.check_if_myopic(X_train, bandwidth=[
        1, 0.1, 0.001, 0.0001], count=min(1000, X_train.shape[0]))["recommended bandwidth"].item(), vgan.subspaces.shape[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    epochs = [5000, 1100, 5000, 2600, 2100, 1600, 1500, 2150, 3000, 5000, 3300]
    proba_p1_array = []
    proba_p2_array = []
    freq_vec = []
    for seed in [777, 1234, 12345, 000, 1000]:
        for i, freq in enumerate([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]):
            proba_p1, proba_p2 = launch_outlier_detection_experiments(freq, [
                LOF()], gen_model_to_use="VMMD", seed=seed,   epochs=epochs[i])

            proba_p1_array.append(proba_p1)
            proba_p2_array.append(proba_p2)
            freq_vec.append(freq)
            pd.DataFrame({"p1": proba_p1_array, "p2": proba_p2_array, "frec": freq_vec}).to_csv(
                f"experiments/Synthetic/table_result.csv")
