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
            nn.LeakyReLU(),
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
            nn.LeakyReLU(),
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = nn.Sequential(
            ConvActBatNorm(1, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            ConvActBatNorm(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            ConvActBatNorm(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            ConvActBatNorm(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Flatten(),
        )

        self.z_mean = nn.Sequential(
            nn.Linear(28 * 28 * 64, 256),
        )
        self.z_log_var = nn.Sequential(
            nn.Linear(28 * 28 * 64, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 28 * 28 * 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(28 * 28 * 64),
            Reshape(-1, 64, 28, 28),
            ConvTActBatNorm(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            ConvTActBatNorm(64, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            ConvTActBatNorm(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            ConvTActBatNorm(32, 1, (3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.z_mean(x), self.z_log_var(x)

    def reparametrize(self, mu, logvar: torch.Tensor):
        std = torch.exp(logvar / 2)
        epsilon = torch.normal(mean=0.0, std=1.0, size=mu.shape, requires_grad=True, device=self.device)
        z = mu + std * epsilon
        return z

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
