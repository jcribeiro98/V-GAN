from pyod.utils.torch_utility import get_activation_by_name
import torch
from torch import nn
from ..models.Generator import upper_softmax
import numpy as np

"""Module to add addtional networks for VGAN

Networks are added as nn.netowrks and then utilized by the od_module to implement inside VGAN
 """


class MLPnet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=[500, 100], n_output=20,
                 activation='ReLU', bias=False, batch_norm=False,
                 skip_connection=False):
        super(MLPnet, self).__init__()
        self.skip_connection = skip_connection
        self.n_output = n_output

        num_layers = len(n_hidden)

        if type(activation) == str:
            activation = [activation] * num_layers
            activation.append(None)

        assert len(activation) == len(
            n_hidden) + 1, 'activation and n_hidden are not matched'

        self.layers = []
        for i in range(num_layers + 1):
            in_channels, out_channels = \
                self.get_in_out_channels(i, num_layers, n_features,
                                         n_hidden, n_output, skip_connection)
            self.layers += [
                LinearBlock(in_channels, out_channels,
                            bias=bias, batch_norm=batch_norm,
                            activation=activation[i],
                            skip_connection=skip_connection if i != num_layers else False)
            ]
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.network(x)
        return x

    @staticmethod
    def get_in_out_channels(i, num_layers, n_features, n_hidden, n_output,
                            skip_connection):
        if skip_connection is False:
            in_channels = n_features if i == 0 else n_hidden[i - 1]
            out_channels = n_output if i == num_layers else n_hidden[i]
        else:
            in_channels = n_features if i == 0 else np.sum(
                n_hidden[:i]) + n_features
            out_channels = n_output if i == num_layers else n_hidden[i]
        return in_channels, out_channels


class LinearBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation='Tanh', bias=False, batch_norm=False,
                 skip_connection=False):
        super(LinearBlock, self).__init__()

        self.skip_connection = skip_connection

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

        if activation is not None:
            # self.act_layer = _instantiate_class("torch.nn.modules.activation", activation)
            self.act_layer = get_activation_by_name(activation)
        else:
            self.act_layer = torch.nn.Identity()

        self.batch_norm = batch_norm
        if batch_norm is True:
            dim = out_channels
            self.bn_layer = torch.nn.BatchNorm1d(dim, affine=bias)

    def forward(self, x):
        x1 = self.linear(x)
        x1 = self.act_layer(x1)

        if self.batch_norm is True:
            x1 = self.bn_layer(x1)

        if self.skip_connection:
            x1 = torch.cat([x, x1], axis=1)

        return x1
