""" Stochastic Variational Inference on MNIST.
"""

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
import fire
import src.datasets as sch_datasets

from base_models import MNISTConvNet, MNISTLinearNet, DemoRegressionNet
from model import SVIModel


def train(model, device, data_loader, optimizer, epoch, M, regressive=False):
    """Training routine.

    Args:
        model (schrodinger.VariationalModel): A variational wrapper over a
            user-defined MLE model.
        device (torch.device): An attribute telling torch on which device to
            perform the computation.
        data_loader (torch.utils.data.DataLoader): Reads and prepares data.
        optimizer (torch.optim.Optimizer): The stochastic gradient descent
            optimization method.
        epoch (int): The current epoch.
    """

    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        # move data to cpu/gpu
        data, target = data.to(device), target.to(device)

        # reset the gradients of the model parameters
        optimizer.zero_grad()

        # get the model predictions by monte carlo
        output = torch.stack([model(data) for _ in range(M)]).mean(0)

        # compute the loss function. For SVI this is actually an ELBO and is
        # of the form NLL_loss + KL(q(phi) | p(theta)) where phi are the
        # variational parameters and theta the model's parameters.
        if regressive:
            loss = (
                len(data_loader) *
                F.mse_loss(
                    output.view(-1),
                    target.view(-1),
                    reduction="mean"
                ) +
                model.get_kl_div()
            )
        else:
            loss = (
                len(data_loader) *
                F.nll_loss(
                    output,
                    target,
                    reduction="mean"
                ) +
                model.get_kl_div()
            )

        # compute gradients
        loss.backward()

        # do an optimization step using the computed gradients
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader, M=50, regressive=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # get the model's prediction using M samples from the posterior
            output = model.eval_forward(data, M)

            if regressive:
                test_loss += F.mse_loss(
                    output.view(-1), target.view(-1), reduction="sum"
                ).item()  # sum up batch loss
                correct = 0
            else:
                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


# pylint: disable=C0330
def main(
    seed=1,
    epochs=10,
    hidden_dim=256,
    batch_size=64,
    lr=0.0005,
    momentum=0.5,
    no_cuda=False,
    M_test=128,
    M_train=16
):
    """ Entry point.
    """
    # pylint: enable=C0330

    torch.manual_seed(seed)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    kwargs = {}
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist", train=True, download=True, transform=transform
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/mnist", train=False, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    model = MNISTConvNet(hidden_dim=hidden_dim).to(device)
    svi_model = SVIModel(model)
    optimizer = optim.Adam(svi_model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train(svi_model, device, train_loader, optimizer, epoch, M_train)
        test(svi_model, device, test_loader, M_test)


def main_reg(
    seed=1,
    epochs=100,
    hidden_dim=30,
    batch_size=4,
    lr=0.01,
    momentum=0.5,
    no_cuda=False,
    M_test=128,
    M_train=16
):
    """ Entry point.
    """
    # pylint: enable=C0330

    torch.manual_seed(seed)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose(
        [lambda x: x.unsqueeze(1), ]
    )
    # kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        sch_datasets.DemoData('sin', is_train=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        sch_datasets.DemoData('sin', is_train=False, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    model = DemoRegressionNet(hidden_dim=hidden_dim).to(device)
    svi_model = SVIModel(model)
    optimizer = optim.Adam(svi_model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train(svi_model, device, train_loader, optimizer, epoch, M_train, True)
        test(svi_model, device, test_loader, M_test, True)


if __name__ == "__main__":
    fire.Fire(main)
    # fire.Fire(main_reg)
