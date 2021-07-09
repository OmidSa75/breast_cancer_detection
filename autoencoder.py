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
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class MSCNN(nn.Module):
    def __init__(self, in_channels, out_channel, features=8):
        super(MSCNN, self).__init__()

        self.conv1 = ConvActBatNorm(in_channels, features, kernel_size=(1, 1), padding=(0, 0))
        self.conv4 = ConvActBatNorm(in_channels, features, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = ConvActBatNorm(in_channels, features, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv = ConvActBatNorm(3 * features, out_channel, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv4(x)
        conv5 = self.conv5(x)
        cat = torch.cat([conv1, conv3, conv5], dim=1)
        return self.pool(self.conv(self.pool(cat)))


class MSTCNN(nn.Module):
    def __init__(self, in_channels, out_channels, feature=8):
        super(MSTCNN, self).__init__()
        self.convt1 = ConvTActBatNorm(in_channels, feature, kernel_size=(1, 1), padding=(0, 0))
        self.convt2 = ConvTActBatNorm(in_channels, feature, kernel_size=(3, 3), padding=(1, 1))
        self.convt3 = ConvTActBatNorm(in_channels, feature, kernel_size=(5, 5), padding=(2, 2))
        self.upsample = nn.Upsample(scale_factor=2)
        self.convt = ConvTActBatNorm(3 * feature, out_channels, kernel_size=(1, 1), padding=(0, 0))

    def forward(self, x):
        convt1 = self.convt1(x)
        convt3 = self.convt2(x)
        convt5 = self.convt3(x)
        cat = torch.cat([convt1, convt3, convt5], dim=1)
        return self.upsample(self.convt(self.upsample(cat)))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mscnn_1 = MSCNN(3, 32, 32)
        self.mscnn_2 = MSCNN(32, 64, 64)
        self.flatten = nn.Flatten()
        self.z_mean = nn.Linear(8 * 8 * 64, 128)
        self.z_log_var = nn.Linear(8 * 8 * 64, 128)

    def forward(self, x):
        x = self.mscnn_1(x)
        x = self.mscnn_2(x)
        x = self.flatten(x)
        log_var = self.z_log_var(x)
        mu = self.z_mean(x)
        std = torch.exp(log_var / 2)
        epsilon = torch.normal(mean=0.0, std=1.0, size=mu.shape, requires_grad=True, device=mu.device)
        z = mu + std * epsilon
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.mstcnn1 = MSTCNN(64, 32, 64)
        self.mstcnn2 = MSTCNN(32, 3, 32)
        self.fc = nn.Sequential(
            nn.Linear(128, 8 * 8 * 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(8 * 8 * 64),
            Reshape(-1, 64, 8, 8)
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.mstcnn1(x)
        x = self.mstcnn2(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "ComplexVAE"

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
