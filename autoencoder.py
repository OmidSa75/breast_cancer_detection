import torch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ConvActBatNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
        super(ConvActBatNorm, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class ConvTActBatNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
        super().__init__()

        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class MSCNN(nn.Module):
    def __init__(self, in_channels, out_channel, features=8):
        super(MSCNN, self).__init__()

        self.conv3 = ConvActBatNorm(in_channels, features, kernel_size=(1, 1), padding=(0, 0))
        self.conv4 = ConvActBatNorm(in_channels, features, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = ConvActBatNorm(in_channels, features, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv = ConvActBatNorm(3 * features, out_channel, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        conv1 = self.conv3(x)
        conv3 = self.conv4(x)
        conv5 = self.conv5(x)
        cat = torch.cat([conv1, conv3, conv5], dim=1)
        return self.pool(self.conv(self.pool(cat)))


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "VAE"
        self.encoder = nn.Sequential(
            ConvActBatNorm(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
            ConvActBatNorm(32, 16, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 16, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
        )

        self.z_mean = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.z_log_var = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 8 * 8 * 16),
            nn.ReLU(),
            nn.BatchNorm1d(8 * 8 * 16),
            Reshape(-1, 16, 8, 8),
            nn.Upsample(scale_factor=2),
            ConvTActBatNorm(16, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            ConvTActBatNorm(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )

        self.logscale = nn.Parameter(torch.Tensor([0.0]))

    def encode(self, x):
        x = self.encoder(x)
        return self.z_mean(x), self.z_log_var(x)

    def reparametrize(self, mu, logvar: torch.Tensor):
        std = torch.exp(logvar / 2)
        try:
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
            return z, std
        except Exception as e:
            print(e)
            print(mu, std)
            assert 'error'

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, std = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, z, std, self.logscale
