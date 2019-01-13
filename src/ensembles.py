""" Ensembles Modules.
"""
from copy import deepcopy

import torch
from torch import nn


def initializer(module):
    """ Callback for resetting a module's weights to Xavier Normal and
        biases to zero.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        module.bias.data.zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        module.bias.data.zero_()


class BootstrappEnsemble(nn.Module):
    def __init__(self, proto, k=5):
        """ COnstructs and anchored ensemble given a prototype network.

        Args:
            nn (nn.Module): Parent object.
            proto (nn.Module): A prototype network that will be used
                to create the ensemble.
            k (int, optional): Defaults to 5. Size of the ensemble.
            data_var (float, optional): Defaults to 0.2. Estimate of the
                data noise.
        """
        super(BootstrappEnsemble, self).__init__()
        self.__ensemble = nn.ModuleList([deepcopy(proto) for _ in range(k)])

        # We are assuming that the prototype network was initialized with
        # Xavier Normal(0, std).
        # TODO: We need to change this and compute μ and Σ priors.
        for model in self.__ensemble:
            model.apply(initializer)

    def forward(self, xs):
        return [model(x) for model, x in zip(self.__ensemble, xs)]

    def parameters(self):
        return [{"params": model.parameters()} for model in self.__ensemble]

    def __len__(self):
        return len(self.__ensemble)
