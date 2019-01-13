import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from datasets import FNS, DemoData
from torch.utils import data
from torch.nn.modules.loss import MSELoss, L1Loss
from torch.optim import Adam
from torch.autograd import Variable
from vadam import Vadam

n_epochs = 10

class FF(nn.Module):
    def __init__(self, h1_size=256, h2_size=128):
        super(FF, self).__init__()
        self.l1 = nn.Linear(1, h1_size)
        self.l2 = nn.Linear(h1_size, h2_size)
        self.l3 = nn.Linear(h2_size, 1)

    def forward(self, x):
        v = F.relu(self.l1(x))
        v = F.relu(self.l2(v))
        v = self.l3(v)
        return v


def train(model, loss, optimizer, train_loader, test_loader):
    for epoch in range(n_epochs):
        for idx, (x, target) in enumerate(train_loader):
            x, target = Variable(x), Variable(target)
            def closure():
                optimizer.zero_grad()
                logits = model(x)
                l = loss(logits, target)
                l.backward()
                return l
            optimizer.step(closure)

    losses = []
    for idx, (x, target) in enumerate(test_loader):
        x, target = Variable(x), Variable(target)
        output = model(x)
        l = loss(output, target)
        losses.append(l.data)
    print(f"Test set loss: {sum(losses)/len(losses)}")


if __name__ == "__main__":
    train_loader = data.DataLoader(DemoData(dset='sin'), batch_size=1)
    test_loader  = data.DataLoader(DemoData(dset='sin', is_train=False), batch_size=1)
    N = len(train_loader)

    model = FF()
    loss = L1Loss()
    optimizer = Vadam(model.parameters(), N, num_samples=20)#, lr=0.0005)
    train(model, loss, optimizer, train_loader, test_loader)


