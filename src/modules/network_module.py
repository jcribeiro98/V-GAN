import torch
from torch import nn
from ..models.Generator import upper_softmax
"""Module to add addtional networks for VGAN

Networks are added as nn.netowrks and then utilized by the od_module to implement inside VGAN
 """
