from collections import OrderedDict

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


def kl_div(post):
    """ Computes the KL divergence between a normal multivariate and a
        diagonal normal distribution.

    Args:
        post (torch.distribution.Normal): A normal distribution.

    Returns:
        torch.tensor: The KL Divergence
    """
    return (post.loc.pow(2) + post.scale.exp() - 1 - post.scale).sum() * 0.5


def reparameterize(mu, logvar):
    """ Implements the reparametrization trick.
    """
    # Not used. You can either use this or the torch.distributions API as we
    # did bellow with `posterior.rsample()` which also implements the
    # reparametrization trick.
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu)
    return eps.mul(std).add_(mu)


class SVIModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self._mle_model = model
        self._posterior = OrderedDict()
        self._prior = OrderedDict()

        self.predictions = None

        for weight_name, weight in model.named_parameters():
            # register the distribution and its parameters
            self._posterior[weight_name] = Normal(
                loc=nn.Parameter(torch.randn_like(weight) * 0.1),
                scale=nn.Parameter(torch.randn_like(weight) * 0.001),
            )
            # register the priors. Actually not needed because we have a
            # special case for computing the KL.
            self._prior[weight_name] = Normal(
                loc=torch.randn_like(weight), scale=torch.randn_like(weight)
            )

    def forward(self, x):
        """ This is a hackish forward in which we are reconstructing the
        forward operations in `self._mle_model`. Ideally this function would
        only:
            1. draw samples from the posterior distribution using the
            reparametrization trick.
            2. perform inference with the **original** model and the sampled
            weights.
        """
        # draw samples from the posterior distribution
        posterior_sample = {
            weight_name: posterior.rsample()
            for weight_name, posterior in self._posterior.items()
        }

        # do a hackish forward through the network
        for layer_name, layer in self._mle_model.named_children():
            if isinstance(layer, nn.Linear):
                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)

                x = F.linear(
                    x,
                    posterior_sample[f"{layer_name}.weight"],
                    posterior_sample[f"{layer_name}.bias"],
                )
            elif isinstance(layer, nn.Conv2d):
                x = F.conv2d(
                    x,
                    posterior_sample[f"{layer_name}.weight"],
                    posterior_sample[f"{layer_name}.bias"],
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                    layer.groups,
                )
            elif isinstance(layer, nn.MaxPool2d):
                x = F.max_pool2d(
                    x,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                    layer.ceil_mode,
                    layer.return_indices
                )
            elif isinstance(layer, nn.LogSoftmax):
                x = F.log_softmax(
                    x,
                    layer.dim
                )
            else:
                raise TypeError("Don't know what type of layer this is.")

            if layer_name not in ["out", "out_activ"]:
                x = torch.relu(x)

        return x

    def eval_forward(self, x, M):
        """ Perform inference using M posterior samples.
        """
        self.predictions = torch.stack([self.forward(x) for _ in range(M)])
        return self.predictions.mean(0)

    def get_kl_div(self):
        """ Return the KL divergence for the complete posterior distribution.
        """
        kls = [kl_div(posterior) for posterior in self._posterior.values()]
        return torch.stack(kls).sum()

    def parameters(self):
        """ Returns the variational parameters of the posterior distribution.
        """
        params = []
        for dist in self._posterior.values():
            params.append(dist.loc)
            params.append(dist.scale)
        return params

    def get_predictive_variance(self, regression=False):
        """ Returns the predictive variance for the result of the last call to
        eval_forward
        """
        probabilities = self.predictions.exp()

        sigma2 = probabilities.var(0)

        if not regression:
            y_hat = probabilities.mean(0).max(1)[1]
            sigma2 = sigma2.gather(1, y_hat.unsqueeze(1))

        return sigma2
