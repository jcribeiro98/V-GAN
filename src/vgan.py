import torch
from collections import defaultdict
from .models.Generator import Generator_big
from .models.Detector import Detector, Encoder, Decoder
import torch_two_sample as tts
from .models.Mmd_loss_constrained import MMDLossConstrained
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import operator
from sklearn.preprocessing import normalize
from typing import Union
from torch.autograd import Variable


class VGAN:
    '''
    VGAN, a Subspace Generation Network.

    Class for the SG method VGAN presented in the paper "Adversarial Subspace Generation for Outlier Detection in High-Dimensional Data"
    This class trains VGAN with kernel learning. The defaults parameters are set as in the paper
    '''

    def __init__(self, batch_size=500, temperature=0, epochs=2000, lr_G=0.007, lr_D=0.007, iternum_d=1, iternum_g=5, momentum=0.99, seed=777, weight_decay=0.04, path_to_directory=None):
        self.storage = locals()
        self.train_history = defaultdict(list)
        self.batch_size = batch_size
        # Experimental. Not used in the experiments found in the paper.
        self.temperature = temperature
        # Experimental. Not used in the experiments found in the paper.
        self.epochs = epochs
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.iternum_d = iternum_d
        self.iternum_g = iternum_g
        self.momentum = momentum
        self.seed = seed
        self.weight_decay = weight_decay
        self.path_to_directory = path_to_directory
        self.generator_optimizer = None
        self.__elm = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.seed = 777

    def __normalize(x, dim=1):
        return x.div(x.norm(2, dim=dim).expand_as(x))

    def __distance(self, x, y, dist):
        """
        Computes distance between corresponding points points in `x` and `y`
        using distance `dist`.
        """
        if dist == 'L2':
            return (x - y).pow(2).mean()
        elif dist == 'L1':
            return (x - y).abs().mean()
        elif dist == 'cos':
            x_n = self.__normalize(x)
            y_n = self.__normalize(y)
            return 2 - (x_n).mul(y_n).mean()
        else:
            assert dist == 'none', 'wtf ?'

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0)

    def __plot_loss(self, path_to_directory, show=False):
        train_history = self.train_history
        plt.style.use('ggplot')
        generator_y = train_history['generator_loss']
        detector_y = train_history['detector_loss']
        x = np.linspace(1, len(generator_y), len(generator_y))
        fig, ax = plt.subplots()
        ax.plot(x, generator_y, color="cornflowerblue",
                label="Generator loss", linewidth=2)
        ax.plot(x, detector_y, color="black",
                label="Detector loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax.legend(loc="upper right")
        plt.savefig(path_to_directory / "train_history.pdf",
                    format="pdf", dpi=1200)

        if show == True:
            print("The show option has been depricated due to lack of utility")

    def get_params(self) -> dict:
        """Gets the parameters of the network as a dictionary
        Returns
            - A dictionary containing the parameters used in the network
        """
        return {'batch size': self.batch_size, 'epochs': self.epochs, 'lr_g': self.lr_G,
                'momentum': self.momentum, 'weight decay': self.weight_decay,
                'batch_size': self.batch_size, 'seed': self.seed,
                'generator optimizer': self.generator_optimizer}

    def model_snapshot(self, path_to_directory=None, run_number=0, show=False):
        ''' Creates an snapshot of the model 

        Saves important information regarding the training of the model
        Args:
            - path_to_directory (Path): Specifies the path to directory (relative to the WD)
            - show (bool): Boolean specifying if a pop-up window should open to show the plot for previsualization.
        '''

        if path_to_directory == None:
            path_to_directory = self.path_to_directory
        path_to_directory = Path(path_to_directory)
        if operator.not_(path_to_directory.exists()):
            os.mkdir(path_to_directory)
        if operator.not_((path_to_directory/"train_history").exists()):
            os.mkdir(path_to_directory / "train_history")

        pd.DataFrame(self.train_history["generator_loss"]).to_csv(
            path_to_directory/'train_history'/f'generator_loss_{run_number}.csv', header=False, index=False)
        if os.path.isfile(path_to_directory/'params.csv') != True:
            pd.DataFrame(self.get_params(), [0]).to_csv(
                path_to_directory / 'params.csv')
        else:
            params = pd.read_csv(path_to_directory / 'params.csv', index_col=0)
            params_new = pd.DataFrame(self.get_params(), [run_number])
            params = params.reindex(params.index.union(params_new.index))
            params.update(params_new)
            params.to_csv(
                path_to_directory / 'params.csv')
        self.__plot_loss(path_to_directory, show=False)

    def load_models(self, path_to_generator, ndims, device: str = None):
        '''Loads models for prediction

        In case that the generator has already been trained, this method allows to load it (and optionally the discriminator) for generating subspaces
        Args:
            - path_to_generator: Path to the generator (has to be stored as a .keras model)
            - path_to_discriminator: Path to the discriminator (has to be stored as a .keras model) (Optional)
        '''
        if device == None:
            device = self.device
        self.generator = Generator_big(
            img_size=ndims, latent_size=max(int(ndims/16), 1)).to(device)
        self.generator.load_state_dict(torch.load(
            path_to_generator, map_location=device))
        self.generator.eval()  # This only works for dropout layers
        self.generator_optimizer = f'Loaded Model from {path_to_generator} with {ndims} dimensions in the latent space'
        self.__latent_size = max(int(ndims/16), 1)

    def get_the_networks(self, ndims: int, latent_size: int, device: str = None) -> tuple:
        """Object function to obtain the networks' architecture

        Args:
            ndims (int): Number of dimensions of the full space
            latent_size (int): Number of dimensions of the latent space
            device (str, optional): CUDA device to mount the networks to. Defaults to None.

        Returns:
            tuple: A tuple containing the generator and the detector of the network (child classes from torch.nn.Module)
        """
        if device == None:
            device = self.device
        generator = Generator_big(
            img_size=ndims, latent_size=latent_size).to(device)
        detector = Detector(latent_size, ndims, Encoder, Decoder).to(device)
        return generator, detector

    def fit(self, X):
        """Fit function for the network 

        Fits the Network with the given dataset X. 
        Args:
            - X: dataset to fit the network with 
        """

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()

        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        # MODEL INTIALIZATION#
        self.__latent_size = latent_size = max(int(X.shape[1]/16), 1)
        ndims = X.shape[1]
        train_size = X.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        device = self.device
        generator, detector = self.get_the_networks(
            ndims, latent_size, device=device)
        generator.apply(self.__weights_init)
        detector.apply(self.__weights_init)

        gen_optimizer = torch.optim.Adadelta(
            generator.parameters(), lr=self.lr_G, weight_decay=self.weight_decay)
        det_optimizer = torch.optim.Adadelta(
            detector.parameters(), lr=self.lr_D, weight_decay=self.weight_decay)
        self.generator_optimizer = gen_optimizer.__class__.__name__
        self.detector_optimizer = det_optimizer.__class__.__name__
        # loss_function =  tts.MMDStatistic(self.batch_size, self.batch_size)
        loss_function = MMDLossConstrained(weight=self.temperature)

        # OPTIMIZATION STUFF
        one = torch.Tensor([1]).to(self.device)
        minusone = one * -1
        minusone = minusone.to(device)
        # DATA LOADER#
        if cuda:
            data_loader = DataLoader(
                X, batch_size=self.batch_size, drop_last=True, pin_memory=cuda, shuffle=True)
        else:  # Uses CUDA if Available, other wise MPS or nothing
            data_loader = DataLoader(
                X, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)
        batch_number = data_loader.__len__()

        # BATCH LOOP#
        iternum_d = 1
        iternum_g = 1
        detector_loss = np.nan
        generator_loss = np.nan
        for epoch in range(self.epochs):
            print(f'\rEpoch {epoch} of {self.epochs}')

            # GET NOISE TENSORS#
            if cuda:
                noise_tensor = torch.cuda.FloatTensor(
                    self.batch_size, latent_size).to(torch.device('cuda'))
            elif mps:
                noise_tensor = torch.mps.Tensor(
                    self.batch_size, latent_size).to(torch.device('mps'))
            else:
                noise_tensor = torch.Tensor(self.batch_size, latent_size)

            # ELM. Not used in the experiments in the paper
            if self.__elm == True:
                for p in detector.encoder.parameters():
                    p.requires_grad = False
            if iternum_d <= self.iternum_d:
                detector_loss = 0
                for batch in tqdm(data_loader, leave=False):
                    # Make sure there is only 1 observation per row.
                    batch = batch.view(self.batch_size, -1)
                    if cuda:
                        # I have no idea why it stopped working in float64 all of a suden >:/
                        batch = batch.to(torch.float32)
                        batch = batch.cuda()
                    elif mps:
                        batch = batch.to(torch.float32).to(
                            # float64 not suported with mps
                            torch.device('mps'))

                    # GET SUBSPACES AND ENCODING-DECODING
                    for p in detector.decoder.parameters():
                        p.requires_grad = True
                    batch_enc, batch_dec = detector(batch)
                    with torch.no_grad():
                        noise_tensor = Variable(noise_tensor.normal_())
                        fake_subspaces = Variable(
                            # Freeze G
                            generator(noise_tensor).clone().detach())
                    projected_batch_enc, projected_batch_dec = detector(
                        fake_subspaces*batch)
                    L2_distance_batch = self.__distance(
                        batch.view(self.batch_size, -1), batch_dec, 'L2')
                    L2_distance_projected_batch = self.__distance(
                        (fake_subspaces*batch).view(self.batch_size, -1), projected_batch_dec, 'L2')

                    # OPTIMIZATION STEP DETECTOR
                    det_optimizer.zero_grad()
                    batch_loss_D = minusone*(loss_function(batch_enc, projected_batch_enc, fake_subspaces) - .1 *
                                             L2_distance_batch - .1*L2_distance_projected_batch)  # Constrained MMD Loss
                    self.bandwidth = loss_function.bandwidth
                    batch_loss_D.backward()
                    det_optimizer.step()
                    detector_loss += float(batch_loss_D.to(
                        'cpu').detach().numpy())/batch_number
                iternum_d += 1
                iternum_g = 1

            elif iternum_g <= self.iternum_g:
                generator_loss = 0
                for batch in tqdm(data_loader, leave=False):
                    # Make sure there is only 1 observation per row.
                    batch = batch.view(self.batch_size, -1)
                    if cuda:
                        batch = batch.cuda()
                        batch = batch.to(torch.float32)
                    elif mps:
                        batch = batch.to(torch.float32).to(
                            # float64 not suported with mps
                            torch.device('mps'))
                    # GET SUBSPACES AND ENCODING-DECODING
                    batch_enc, batch_dec = detector(batch)
                    noise_tensor = Variable(noise_tensor.normal_())
                    fake_subspaces = Variable(
                        generator(noise_tensor))  # Unfreeze G
                    fake_subspaces.requires_grad = True
                    projected_batch_enc, projected_batch_dec = detector(
                        fake_subspaces*batch)
                    L2_distance_batch = self.__distance(
                        batch.view(self.batch_size, -1), batch_dec, 'L2')
                    L2_distance_projected_batch = self.__distance(
                        (fake_subspaces*batch).view(self.batch_size, -1), projected_batch_dec, 'L2')

                    # OPTIMIZATION STEP GENERATOR
                    for p in detector.parameters():
                        p.requires_grad = False
                    gen_optimizer.zero_grad()

                    batch_loss_G = loss_function(
                        batch_enc, projected_batch_enc, fake_subspaces)  # Constrained MMD Loss
                    self.bandwidth = loss_function.bandwidth
                    batch_loss_G.backward()
                    gen_optimizer.step()
                    generator_loss += float(batch_loss_G.to(
                        'cpu').detach().numpy())/batch_number
                iternum_g += 1
                if iternum_g > self.iternum_g:
                    iternum_d = 1

            print(f"Average loss in the epoch Generator: {generator_loss}")
            print(f"Average loss in the epoch Detector: {detector_loss}")
            self.train_history["generator_loss"].append(generator_loss)
            self.train_history["detector_loss"].append(detector_loss)

        if not self.path_to_directory == None:
            path_to_directory = Path(self.path_to_directory)
            if operator.not_(path_to_directory.exists()):
                os.mkdir(path_to_directory)
                if operator.not_(Path(path_to_directory/'models').exists()):
                    os.mkdir(path_to_directory / 'models')
            run_number = int(len(os.listdir(path_to_directory/'models'))/2)
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'generator_{run_number}.pt')
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'detector_{run_number}.pt')
            self.model_snapshot(path_to_directory, run_number, show=True)

        self.generator = generator
        self.detector = detector

    def generate_subspaces(self, nsubs) -> torch.Tensor:
        """Generates subspaces

        Calls for nsubs foward passes of the network
        Args:
            - nsubs: number of subspaces to generate
        Return:
            - u: Tensor of subspaces (in cpu)
        """
        noise_tensor = torch.Tensor(nsubs, self.__latent_size).to('cpu')
        if not self.seed == None:
            torch.manual_seed(self.seed)
        noise_tensor.normal_()
        u = self.generator(noise_tensor.to(self.device))
        u = torch.greater_equal(u, 1/u.shape[1])
        return u

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


class VGAN_no_kl:
    '''
    VGAN, a Subspace Generation Network.

    Class for the SG method VGAN presented in the paper "Adversarial Subspace Generation for Outlier Detection in High-Dimensional Data"
    This class trains VGAN without kernel learning. The defaults parameters are set as in the paper.
    '''

    def __init__(self, batch_size=500, epochs=2000, lr=0.007, momentum=0.99, seed=777, weight_decay=0.04, path_to_directory=None):
        self.storage = locals()
        self.train_history = defaultdict(list)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        self.weight_decay = weight_decay
        self.path_to_directory = path_to_directory
        self.generator_optimizer = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available(
        ) else 'mps:0' if torch.backends.mps.is_available() else 'cpu')

    def __plot_loss(self, path_to_directory, show=False):
        train_history = self.train_history
        plt.style.use('ggplot')
        generator_y = train_history['generator_loss']
        x = np.linspace(1, len(generator_y), len(generator_y))
        fig, ax = plt.subplots()
        ax.plot(x, generator_y, color="cornflowerblue",
                label="Generator loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        ax.legend(loc="upper right")
        plt.savefig(path_to_directory / "train_history.pdf",
                    format="pdf", dpi=1200)

        if show == True:
            print("The show option has been depricated due to lack of utility")

    def get_params(self) -> dict:
        return {'batch size': self.batch_size, 'epochs': self.epochs, 'lr_g': self.lr,
                'momentum': self.momentum, 'weight decay': self.weight_decay,
                'batch_size': self.batch_size, 'seed': self.seed,
                'generator optimizer': self.generator_optimizer}

    def model_snapshot(self, path_to_directory=None, run_number=0, show=False):
        ''' Creates an snapshot of the model

        Saves important information regarding the training of the model
        Args:
            - path_to_directory (Path): Specifies the path to directory (relative to the WD)
            - show (bool): Boolean specifying if a pop-up window should open to show the plot for previsualization.
        '''

        if path_to_directory == None:
            path_to_directory = self.path_to_directory
        path_to_directory = Path(path_to_directory)
        if operator.not_(path_to_directory.exists()):
            os.mkdir(path_to_directory)
        if operator.not_((path_to_directory/"train_history").exists()):
            os.mkdir(path_to_directory / "train_history")

        pd.DataFrame(self.train_history["generator_loss"]).to_csv(
            path_to_directory/'train_history'/f'generator_loss_{run_number}.csv', header=False, index=False)
        if os.path.isfile(path_to_directory/'params.csv') != True:
            pd.DataFrame(self.get_params(), [0]).to_csv(
                path_to_directory / 'params.csv')
        else:
            params = pd.read_csv(path_to_directory / 'params.csv', index_col=0)
            params_new = pd.DataFrame(self.get_params(), [run_number])
            params = params.reindex(params.index.union(params_new.index))
            params.update(params_new)
            params.to_csv(
                path_to_directory / 'params.csv')
        self.__plot_loss(path_to_directory, show)

    def load_models(self, path_to_generator, ndims, device: str = None):
        '''Loads models for prediction

        In case that the generator has already been trained, this method allows to load it (and optionally the discriminator) for generating subspaces
        Args:
            - path_to_generator: Path to the generator (has to be stored as a .keras model)
            - path_to_discriminator: Path to the discriminator (has to be stored as a .keras model) (Optional)
        '''
        if device == None:
            device = self.device
        self.generator = Generator_big(
            img_size=ndims, latent_size=max(int(ndims/16), 1)).to(device)
        self.generator.load_state_dict(torch.load(
            path_to_generator, map_location=device))
        self.generator.eval()  # This only works for dropout layers
        self.generator_optimizer = f'Loaded Model from {path_to_generator} with {ndims} dimensions in the latent space'
        self.__latent_size = max(int(ndims/16), 1)

    def get_the_networks(self, ndims: int, latent_size: int, device: str = None) -> Generator_big:
        """Object function to obtain the networks' architecture

        Args:
            ndims (int): Number of dimensions of the full space
            latent_size (int): Number of dimensions of the latent space
            device (str, optional): CUDA device to mount the networks to. Defaults to None.

        Returns:
            generator: A generator model (child class from torch.nn.Module)
        """
        if device == None:
            device = self.device
        generator = Generator_big(
            img_size=ndims, latent_size=latent_size).to(device)
        return generator

    def fit(self, X):

        cuda = torch.cuda.is_available()
        mps = torch.backends.mps.is_available()

        torch.manual_seed(self.seed)
        if cuda:
            torch.cuda.manual_seed(self.seed)
        elif mps:
            torch.mps.manual_seed(self.seed)

        # MODEL INTIALIZATION#
        epochs = self.epochs
        self.__latent_size = latent_size = max(int(X.shape[1]/16), 1)
        ndims = X.shape[1]
        train_size = X.shape[0]
        self.batch_size = min(self.batch_size, train_size)

        device = self.device
        generator = self.get_the_networks(
            ndims, latent_size, device=device)
        optimizer = torch.optim.Adadelta(
            generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.generator_optimizer = optimizer.__class__.__name__
        # loss_function =  tts.MMDStatistic(self.batch_size, self.batch_size)
        loss_function = MMDLossConstrained(weight=10)

        for epoch in range(epochs):
            print(f'\rEpoch {epoch} of {epochs}')
            generator_loss = 0

            # DATA LOADER#
            if cuda:
                data_loader = DataLoader(
                    X, batch_size=self.batch_size, drop_last=True, pin_memory=cuda, shuffle=True)
            else:  # Uses CUDA if Available, other wise MPS or nothing
                data_loader = DataLoader(
                    X, batch_size=self.batch_size, drop_last=True, pin_memory=mps, shuffle=True)
            batch_number = data_loader.__len__()

            # GET NOISE TENSORS#
            if cuda:  # Need to open this if statement as the Tensor function has to be called from diferent modules depending of the device
                noise_tensor = torch.cuda.FloatTensor(
                    self.batch_size, latent_size).to(torch.device('cuda'))
            elif mps:
                noise_tensor = torch.mps.Tensor(
                    self.batch_size, latent_size).to(torch.device('mps'))
            else:
                noise_tensor = torch.Tensor(self.batch_size, latent_size)

            # BATCH LOOP#
            for batch in tqdm(data_loader, leave=False):
                # Make sure there is only 1 observation per row.
                batch = batch.view(self.batch_size, -1)
                if cuda:
                    # It stopped working for some f****** reason
                    batch = batch.to(torch.float32)
                    batch = batch.cuda()

                elif mps:
                    batch = batch.to(torch.float32).to(
                        torch.device('mps'))  # float64 not suported with mps

                # SAMPLE NOISE#
                noise_tensor.normal_()

                # OPTIMIZATION STEP#
                optimizer.zero_grad()
                fake_subspaces = generator(noise_tensor)
                batch_loss = loss_function(
                    batch, fake_subspaces*batch, fake_subspaces)
                self.bandwidth = loss_function.bandwidth
                batch_loss.backward()
                optimizer.step()
                generator_loss += float(batch_loss.to(
                    'cpu').detach().numpy())/batch_number

            print(f"Average loss in the epoch: {generator_loss}")
            self.train_history["generator_loss"].append(generator_loss)

        if not self.path_to_directory == None:
            path_to_directory = Path(self.path_to_directory)
            if operator.not_(path_to_directory.exists()):
                os.mkdir(path_to_directory)
                if operator.not_(Path(path_to_directory/'models').exists()):
                    os.mkdir(path_to_directory / 'models')
            run_number = int(len(os.listdir(path_to_directory/'models')))
            torch.save(generator.state_dict(),
                       path_to_directory/'models'/f'generator_{run_number}.pt')
            self.model_snapshot(path_to_directory, run_number, show=True)

        self.generator = generator

    def generate_subspaces(self, nsubs):
        # Need to load in cpu as mps Tensor module doesn't properly fix the seed
        noise_tensor = torch.Tensor(nsubs, self.__latent_size).to('cpu')
        if not self.seed == None:
            torch.manual_seed(self.seed)
        noise_tensor.normal_()
        u = self.generator(noise_tensor.to(self.device))
        u = torch.greater_equal(u, 1/u.shape[1])
        return u

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
