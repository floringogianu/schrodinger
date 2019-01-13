""" Bootstrapped Ensembles example.
"""
import sys

import torch
from torch import nn
from torch import optim
from torch.utils import data

sys.path.append('')
# pylint: disable=C0413
from src.datasets import DemoData, BootstrappDataset, boot_collate
from src.ensembles import BootstrappEnsemble
from src.losses import EnsembleLoss


def train(epoch, loader, model, optimizer, loss_fn):

    epoch_loss = []

    for idx, (xs, targets) in enumerate(loader):

        optimizer.zero_grad()

        ys = model(xs)  # returns a list of batches (a batch per model)
        loss = loss_fn(ys, targets)
        loss.backward()

        optimizer.step()
        epoch_loss.append(loss.detach())

    epoch_loss = torch.stack(epoch_loss).mean()
    print(f"{epoch:3d}, loss={epoch_loss:5.2f}.")


def main(epochs=1000, k=10, batch_size=5):

    train_dset = BootstrappDataset(DemoData(dset="double_sin"), k=k)
    train_loader = data.DataLoader(
        train_dset, batch_size=batch_size, collate_fn=boot_collate
    )

    model = BootstrappEnsemble(
        nn.Sequential(
            nn.Linear(1, 20, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(20, 1, bias=True),
        ),
        k=k,
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = EnsembleLoss(nn.MSELoss(), k=k)

    for epoch in range(1, epochs+1):
        train(epoch, train_loader, model, optimizer, loss_fn)
        

if __name__ == "__main__":
    main()
