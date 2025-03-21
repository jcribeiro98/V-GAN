from sel_suod.models.base import sel_SUOD
import numpy as np
from src.vgan import VGAN, VGAN_no_kl
from ..models.Detector import Detector, Encoder, Decoder
from ..models.Generator import Generator_big, Generator
import torch_two_sample as tts
from sklearn.preprocessing import normalize
import torch
from ..models.Mmd_loss_constrained import MMDLossConstrained
import pandas as pd
from typing import Union
import logging
logger = logging.getLogger(__name__)


class VGAN(VGAN):
    def get_the_networks(self, ndims, latent_size, device=None):
        if device == None:
            device = self.device
        generator = Generator_big(
            img_size=ndims, latent_size=latent_size).to(device)
        detector = Detector(latent_size, ndims, Encoder, Decoder).to(device)
        return generator, detector

    def approx_subspace_dist(self, subspace_count=500, add_leftover_features=False):
        u = self.generate_subspaces(subspace_count)
        unique_subspaces, proba = np.unique(
            np.array(u.to('cpu')), axis=0, return_counts=True)
        if (unique_subspaces.sum(axis=0) < 1).sum() != 0 and add_leftover_features:
            unique_subspaces = np.append(
                unique_subspaces, [unique_subspaces.sum(axis=0) < 1], axis=0)
            proba = np.append(proba/proba.sum(), 1)

        self.subspaces = unique_subspaces
        self.proba = proba/proba.sum()

    def check_if_myopic(self,  x_data: np.array, bandwidth: Union[float, np.array] = 0.01, count=500) -> pd.DataFrame:
        """Two sample test for myopicity

        Launches the two sample test presented in the paper for checking the myopicity.

        Args:
            x_data (np.array): Data to check the myopicity of.
            bandwidth (float | np.array, optional): Bandwidth used in the GOF tests using the MMD. This method always runs
            the recommended bandwidth alongside this optional one. Defaults to 0.01.
            count (int, optional): Number of samples used to approximate the MMD. Defaults to 500.

        Returns:
            pd.DataFrame: DataFrame containing the p.value of the test with all the different bandwidths.
        """
        assert count <= x_data.shape[0], "Selected 'count' is greater than the number of samples in the dataset"
        results = []

        x_data = normalize(x_data, axis=0)
        x_sample = torch.Tensor(pd.DataFrame(
            x_data).sample(count).to_numpy()).to(self.device)
        u_subspaces = self.generate_subspaces(count)
        ux_sample = u_subspaces * \
            torch.Tensor(x_sample).to(self.device) + \
            torch.mean(x_sample, dim=0)*(~u_subspaces)
        if type(bandwidth) == float:
            bandwidth = [bandwidth]

        if not hasattr(self, 'bandwidth'):
            mmd_loss = MMDLossConstrained(0)
            mmd_loss.forward(
                x_sample, ux_sample, u_subspaces*1)
            self.bandwidth = mmd_loss.bandwidth

        bandwidth.sort()
        for bw in bandwidth:
            mmd = tts.MMDStatistic(count, count)
            _, distances = mmd(x_sample, ux_sample, alphas=[
                bw], ret_matrix=True)
            results.append(mmd.pval(distances))

        bw = self.bandwidth.item()
        mmd = tts.MMDStatistic(count, count)
        _, distances = mmd(x_sample, ux_sample, alphas=[
            bw], ret_matrix=True)
        results.append(mmd.pval(distances))

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])


class VGAN_no_kl(VGAN_no_kl):
    def approx_subspace_dist(self, subspace_count=500, add_leftover_features=False):
        u = self.generate_subspaces(subspace_count)
        unique_subspaces, proba = np.unique(
            np.array(u.to('cpu')), axis=0, return_counts=True)
        if (unique_subspaces.sum(axis=0) < 1).sum() != 0 and add_leftover_features:
            unique_subspaces = np.append(
                unique_subspaces, [unique_subspaces.sum(axis=0) < 1], axis=0)
            proba = np.append(proba/proba.sum(), 1)

        self.subspaces = unique_subspaces
        self.proba = proba/proba.sum()

    def check_if_myopic(self,  x_data: np.array, bandwidth: Union[float, np.array] = 0.01, count=500) -> pd.DataFrame:
        """Two sample test for myopicity

        Launches the two sample test presented in the paper for checking the myopicity.

        Args:
            x_data (np.array): Data to check the myopicity of.
            bandwidth (float | np.array, optional): Bandwidth used in the GOF tests using the MMD. This method always runs
            the recommended bandwidth alongside this optional one. Defaults to 0.01.
            count (int, optional): Number of samples used to approximate the MMD. Defaults to 500.

        Returns:
            pd.DataFrame: DataFrame containing the p.value of the test with all the different bandwidths.
        """
        assert count <= x_data.shape[0], "Selected 'count' is greater than the number of samples in the dataset"
        results = []

        x_data = normalize(x_data, axis=0)
        x_sample = torch.mps.Tensor(pd.DataFrame(
            x_data).sample(count).to_numpy()).to('mps:0')
        u_subspaces = self.generate_subspaces(count)
        ux_sample = u_subspaces * \
            torch.mps.Tensor(x_sample).to(self.device) + \
            torch.mean(x_sample, dim=0)*(~u_subspaces)
        if type(bandwidth) == float:
            bandwidth = [bandwidth]

        if not hasattr(self, 'bandwidth'):
            mmd_loss = MMDLossConstrained(0)
            mmd_loss.forward(
                x_sample, ux_sample, u_subspaces*1)
            self.bandwidth = mmd_loss.bandwidth

        bandwidth.sort()
        for bw in bandwidth:
            mmd = tts.MMDStatistic(count, count)
            _, distances = mmd(x_sample, ux_sample, alphas=[
                bw], ret_matrix=True)
            results.append(mmd.pval(distances))

        bw = self.bandwidth.item()
        mmd = tts.MMDStatistic(count, count)
        _, distances = mmd(x_sample, ux_sample, alphas=[
            bw], ret_matrix=True)
        results.append(mmd.pval(distances))

        bandwidth.append("recommended bandwidth")
        return pd.DataFrame([results], columns=bandwidth, index=["p-val"])
