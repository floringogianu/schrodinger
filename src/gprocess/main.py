import fire
import math
import torch
import os,sys,inspect
from torch.utils import data
from torch.optim import Adam
import gpytorch
from matplotlib import pyplot as plt


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from datasets import DemoData
from model import ExactGPModel
    

def main(dset="sin"):
    #prepare data
    loader = data.DataLoader(DemoData(dset=dset), batch_size=1)
    d = [i for i in list(loader)]
    train_x = torch.squeeze(torch.stack([x[0] for x in d], dim=1), dim=0)
    train_y = torch.squeeze(torch.stack([y[1] for y in d], dim=1), dim=0)

    # initialize 
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    model.train()
    likelihood.train()
    optimizer = Adam([{'params': model.parameters()},], lr=0.1)
    # mll - marginal loss likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    no_steps = 100
    for i in range(no_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        print('step %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
            i + 1, no_steps, loss.item(),
            model.covar_module.base_kernel.log_lengthscale.item(),
            model.likelihood.log_noise.item()
        ))
        optimizer.step()

    model.eval()
    likelihood.eval()

    x_min, x_max = (min(train_x), max(train_x))

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(x_min - 1, x_max + 1, 51)
        observed_pred = likelihood(model(test_x))

    # plotting
    with torch.no_grad():
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
        lower, upper = observed_pred.confidence_region()
        ax.plot(train_x.numpy(), train_y.numpy(), 'k.')
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    
    ax.set_ylim([-2, 4])
    plt.xlabel('X (independent variable)')
    plt.ylabel('Y (dependent variable)')
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)