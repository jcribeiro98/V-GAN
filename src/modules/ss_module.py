"""File containing all of the subspace selection methods. Each method is implemented in its own class, inheriting
basic properties from the base class.
This base class has to store a:
     .subsapce: type np.array[list[bool]]. np.array of subspaces. Each subsapce is a list of boolean indiciating which feature is contained in the subsapce
     .metric: type list[int]. List of a metric/quality associated to each subspace. It will get inizialized by ones. If not used/needed, leave it like that
"""
import random
import src.modules.bin.main_hics as hics
import numpy as np
import pandas as pd
import os
from src.modules.tools import numeric_to_boolean
from pathlib import Path


class BaseSubspaceSelector:
    def __init__(self) -> None:
        self.subspaces = None
        self.metric = None
        self.trained = False
        pass

    def fit(self, X_train):
        self.trained = True


class HiCS(BaseSubspaceSelector):
    """Class for the HiCS method

    Inherits:
        BaseSubspaceSelector
    """

    def __init__(self, numRuns=100,
                 numCandidates=500, maxOutputSpaces=1000, alpha=0.1,
                 silent=False):
        super().__init__()
        # We need to load the data directly in Nim for this to work, we can not pass it as a numpy array.
        self.numRuns = numRuns
        self.numCandidates = numCandidates
        self.maxOutputSpaces = maxOutputSpaces
        self.alpha = alpha
        self.__onlySubspace = []
        self.silent = silent

    def __calculate_the_subspaces(self, X_train: np.ndarray):

        # Nim has troubles dealing with the numpy array direclty, so we store it first as a csv for it to read it
        np.savetxt(Path("src/modules/bin/data.csv"), X_train, delimiter=";")
        results = hics.launch(csvIn="src/modules/bin/data.csv", hasHeader=False,
                              numRuns=self.numRuns, numCandidates=self.numCandidates,
                              maxOutputSpaces=self.maxOutputSpaces, alpha=self.alpha,
                              onlySubspace=self._HiCS__onlySubspace)
        os.remove(Path("src/modules/bin/data.csv"))

        subspaces = []
        for element in results:
            subspace = []
            for item in element[1]['data']:
                if item[1] != 0:
                    subspace.append(item[1])
            subspaces.append(subspace)
        contrast = []
        for element in results:
            contrast.append(element[0])

        return contrast, numeric_to_boolean(subspaces, X_train.shape[1])

    def fit(self, X_train):
        """Fits the HiCS method

        Args:
            X_train (np.array): Numpy array to extract the subspaces from
        """

        self.metric, self.subspaces = self.__calculate_the_subspaces(X_train)
        self.trained = True


# class CLIQUE(BaseSubspaceSelector):
