from sel_suod.models.base import sel_SUOD
import numpy as np
from src.vgan import VGAN
from src.vmmd import VMMD


class VGAN(VGAN):

    def approx_subspace_dist(self, subspace_count=500):
        self.__seed = 777
        u = self.generate_subspaces(subspace_count)
        unique_subspaces, proba = np.unique(
            np.array(u.to('cpu')), axis=0, return_counts=True)
        self.subspaces = unique_subspaces
        self.proba = proba


class VMMD(VMMD):

    def approx_subspace_dist(self, subspace_count=500):
        u = self.generate_subspaces(subspace_count)
        unique_subspaces, proba = np.unique(
            np.array(u.to('cpu')), axis=0, return_counts=True)
        self.subspaces = unique_subspaces
        self.proba = proba
