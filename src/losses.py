""" Special Loss objects.
"""
import torch


class EnsembleLoss:
    def __init__(self, loss_fn, k=5):
        self.__loss_fn = loss_fn
        self.__k = k

    def __call__(self, ys, targets):
        losses = [self.__loss_fn(y, t) for y, t in zip(ys, targets)]
        return torch.stack(losses).sum()
