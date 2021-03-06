import torch
from torch import nn


def gaussian_likelihood(recon_x, logscale, x):
    scale = torch.exp(logscale)
    mean = recon_x
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))


def kl_divergence(z, mu, std):
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum((1,))
    return kl


class VAEClsLoss(nn.Module):
    def __init__(self):
        super(VAEClsLoss, self).__init__()
        self.recon_loss = nn.MSELoss()

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = self.recon_loss(x, recon_x)
        kl_loss = (-0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar)).sum(dim=1)).mean(dim=0)
        total_loss = recon_loss * 2048 + kl_loss
        return total_loss
