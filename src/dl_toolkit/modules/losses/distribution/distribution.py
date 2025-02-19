import numpy as np
import torch


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(AbstractDistribution):
    def __init__(self, parameters, deterministic=False, generator=None):
        self.parameters = parameters
        if isinstance(parameters, (tuple, list)):
            self.mean, self.logvar = parameters
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        assert self.mean.shape == self.logvar.shape, f"{self.mean.shape} != {self.logvar.shape}"
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.generator = generator
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn(
            self.mean.shape,
            device=self.mean.device,
            generator=self.generator,
            dtype=self.mean.dtype,
        )
        return x

    def mode(self):
        return self.mean

    def kl(self, other=None, dim=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.0]).type_as(self.mean)

        # Important note: KL divergence must be summed over all dimensions except the batch dimension
        # because the KL divergence is defined for the single sample
        # Refer to https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/distributions/distributions.py#L24
        # And https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/losses/contperceptual.py#L60
        # To get loss for batch we should get average over batch and sum over all other dimensions
        # You can refer for explanation here https://arxiv.org/pdf/2309.13160
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dim)
                )
            else:
                return 0.5 * torch.mean(
                    torch.sum(torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=dim)
                )

    def nll(self, sample, dim=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.0]).type_as(self.mean)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.mean(
            torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
                      dim=dim)
        )

