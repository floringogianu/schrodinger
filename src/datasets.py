""" Simple regression Datasets used for visualization.
"""
import torch
from torch.utils import data
from torch.distributions import Normal, Uniform


def double_sin(x, α=4, β=13, μ=0, σ=0.03, noisy=True):
    """ Toy function in: https://arxiv.org/pdf/1602.04621.pdf, Fig. 10
    """
    w = 0
    if noisy:
        w = Normal(μ, σ).sample((x.shape[0],))
    return x + torch.sin(α * (x + w)) + torch.sin(β * (x + w)) + w


def sin(x, β=0.2, noisy=True):
    """ Toy function in Bishop, Pattern Recognition... , somewhere in
    introduction.
    """
    if noisy:
        return torch.sin(x) + Normal(0, β).sample((x.shape[0],))


def tricky_line(x):
    """ Toy function in: https://arxiv.org/pdf/1806.03335.pdf, Fig. 13
    """
    y = torch.zeros_like(x)
    y[-1] = 5
    return y


# each entry is [fn, x_train, x_test]
FNS = {
    "double_sin": [
        double_sin,
        torch.cat(
            (Uniform(0, 0.6).sample((24,)), Uniform(0.8, 1).sample((11,)))
        ),
        Uniform(0, 1.1).sample((35,)),
    ],
    "sin": [
        sin,
        torch.cat(
            (Uniform(0.5, 2.3).sample((24,)), Uniform(4.0, 6.0).sample((11,)))
        ),
        Uniform(0, 6.5).sample((11,)),
    ],
    "tricky_line": [
        tricky_line,
        torch.linspace(-0.9, 0.9, 10),
        torch.linspace(-0.9, 0.9, 10),
    ],
}


class DemoData(data.Dataset):
    def __init__(self, dset="sin", is_train=True, transform=None):
        """ Returns a dataset based on a nice looking function.

        Args:
            data (data.Dataset): PyTorch parent class
            dset (str, optional): Defaults to "sin". Dataset name, keys in FNS.
            is_train (bool, optional): Defaults to True. Train or test
            transform (function, optional): Defaults to None. Callaback to a
                function that further transforms the data, for example
                a radial basis function or a polynomial feature extractor.
        """

        self.__train = is_train
        fn, x_train, x_test = FNS[dset]
        if is_train:
            self.__data = x_train
            self.__targets = fn(x_train)
        else:
            self.__data = x_test
            self.__targets = fn(x_test)

        if transform is not None:
            self.__data = transform(self.__data)

    def __getitem__(self, index):
        x, target = self.__data[index], self.__targets[index]
        return x, target

    def __len__(self):
        return self.__data.shape[0]


class BootstrappDataset(data.Dataset):
    def __init__(self, dset, k=5):
        N = len(dset)
        self.__k = k
        self.__original = dset
        self.__masks = [torch.randint(0, N, (N,)) for _ in range(k)]

    def __getitem__(self, index):
        samples = [self.__original[mask[index]] for mask in self.__masks]
        return samples

    def __len__(self):
        return len(self.__original)


def boot_collate(samples):
    features_batches = [
        torch.tensor([[el[0]] for el in batch]) for batch in zip(*samples)
    ]
    target_batches = [
        torch.tensor([[el[1]] for el in batch]) for batch in zip(*samples)
    ]
    # return list(zip(features_batches, target_batches))
    return features_batches, target_batches


def main():
    for dset in FNS.keys():
        loader = data.DataLoader(DemoData(dset=dset), batch_size=1)
        print(f"\nLoading dataset {dset.upper()}.")
        for idx, (x, target) in enumerate(loader):
            print(idx, x, target)
        print("-" * 10)

    k = 3
    bsz = 2
    print(f"\nBootstrappDataset  k={k}  bsz={bsz}---------")
    dset = DemoData(dset="double_sin")
    boot_dset = BootstrappDataset(dset, k=k)
    boot_loader = data.DataLoader(
        boot_dset, batch_size=bsz, collate_fn=boot_collate
    )

    for idx, (xs, targets) in enumerate(boot_loader):
        print("batch: ", idx)
        print("  X:      ", xs)
        print("  target: ", targets)


if __name__ == "__main__":
    main()
