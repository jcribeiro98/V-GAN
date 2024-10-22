from src.modules.od_module import VGAN, VMMD
import numpy as np
from pathlib import Path
import pandas as pd
from src.modules.synthetic_module import generate_data_3d
import logging
import src.modules.ss_module as ss
logger = logging.getLogger(__name__)


def launch_extraction_experiments(freq: int, base_estimators: list, epochs: int = 3000,
                                  temperature: float = 0, seed: int = 777, gen_model_to_use: str = "VMMD") -> dict:
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


def launch_extraction_experiments_hics(freq: int, epochs: int = 3000,
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

    hics = ss.HiCS()
    hics.fit(X_train)
    hics.metric = (hics.metric-np.min(hics.metric)) / \
        (np.max(hics.metric)-np.min(hics.metric)
         )  # This ensures that the final share will be positive
    hics.metric = hics.metric/hics.metric.sum()
    print(pd.DataFrame(hics.subspaces, hics.metric))
    proba_p2 = hics.metric[[hics.subspaces[i].tolist() in [[True, False, True], [
        False, False, True], [False, True, True]] for i in range(hics.subspaces.__len__())]].sum()
    proba_p1 = hics.metric[[hics.subspaces[i].tolist(
    ) in [[True, True, False]] for i in range(hics.subspaces.__len__())]].sum()

    return proba_p1, proba_p2


def launch_extraction_experiments_gmd(freq: int, epochs: int = 3000,
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

    gmd = ss.GMD()
    gmd.fit(X_train)
    gmd.metric = (gmd.metric-np.min(gmd.metric)) / \
        (np.max(gmd.metric)-np.min(gmd.metric)
         )  # This ensures that the final share will be positive
    gmd.metric = gmd.metric/gmd.metric.sum()
    print(pd.DataFrame(gmd.subspaces, gmd.metric))
    proba_p2 = gmd.metric[[gmd.subspaces[i].tolist() in [[True, False, True], [
        False, False, True], [False, True, True], [True, True, True]] for i in range(gmd.subspaces.__len__())]].sum()
    proba_p1 = gmd.metric[[gmd.subspaces[i].tolist(
    ) in [[True, True, False]] for i in range(gmd.subspaces.__len__())]].sum()

    return proba_p1, proba_p2


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    epochs = [5000, 1100, 5000, 2600, 2100, 1600, 1500, 2150, 3000, 5000, 3300]
    proba_p1_array = []
    proba_p2_array = []
    freq_vec = []
    for seed in [777, 1234, 12345, 000, 1000]:
        for i, freq in enumerate([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]):
            proba_p1, proba_p2 = launch_extraction_experiments_gmd(
                freq, seed=seed,   epochs=epochs[i])

            proba_p1_array.append(proba_p1)
            proba_p2_array.append(proba_p2)
            freq_vec.append(freq)
            pd.DataFrame({"p1": proba_p1_array, "p2": proba_p2_array, "frac": freq_vec}).to_csv(
                f"experiments/Synthetic/table_result_gmd.csv")
