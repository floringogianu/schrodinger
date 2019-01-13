import torch.nn as nn
import torch.nn.functional as F


class MNISTLinearNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.maxp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxp2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(320, hidden_dim)
        self.out = nn.Linear(hidden_dim, 10)
        self.out_activ = nn.LogSoftmax(1)

    def forward(self, x):
        x = F.relu(self.maxp1(self.conv1(x), 2))
        x = F.relu(self.maxp2(self.conv2(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return self.out_activ(x)


class MNISTConvNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.out = nn.Linear(hidden_dim, 10)
        self.out_activ = nn.LogSoftmax(1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return self.out_activ(x)


class DemoRegressionNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
