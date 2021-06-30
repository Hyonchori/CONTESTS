import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1, 32, (2, 3, 3), stride=2, padding=(0, 1, 1))
        self.conv3d_2 = nn.Conv3d(32, 64, (2, 3, 3), stride=2, padding=(0, 1, 1))
        self.conv3d_3 = nn.Conv3d(64, 128, (3, 3, 3), stride=2, padding=(0, 1, 1))
        self.conv3ds = [
            self.conv3d_1,
            self.conv3d_2,
            self.conv3d_3
        ]
        self.conv2d_1 = nn.Conv2d(128, 256, 3, stride=2, padding=(1, 1))
        self.conv2d_2 = nn.Conv2d(256, 512, 3, stride=2, padding=(1, 1))
        self.conv2d_3 = nn.Conv2d(512, 1024, 3, stride=2, padding=(1, 1))
        self.conv2ds = [
            self.conv2d_1,
            self.conv2d_2,
            self.conv2d_3
        ]
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dropout(x)
        skips = []
        for conv in self.conv3ds:
            x = conv(x)
            x = self.relu(x)
            skips.append(x)

        shape = x.size()
        x = x.view(shape[0], shape[1], shape[3], shape[4])
        for conv in self.conv2ds:
            x = conv(x)
            x = self.relu(x)
            skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv2d_1 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=(1, 1), output_padding=(1, 1))
        self.deconv2d_2 = nn.ConvTranspose2d(1024, 256, 3, stride=2, padding=(1, 1), output_padding=(1, 0))
        self.deconv2d_3 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=(1, 1), output_padding=(1, 1))
        self.deconv2ds = [
            self.deconv2d_1,
            self.deconv2d_2,
            self.deconv2d_3
        ]
        self.deconv3d_1 = nn.ConvTranspose3d(256, 64, 3, stride=2, padding=(0, 1, 1), output_padding=(0, 1, 1))
        self.deconv3d_2 = nn.ConvTranspose3d(128, 32, 3, stride=2, padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv3d_3 = nn.ConvTranspose3d(64, 16, 3, stride=2, padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv3ds = [
            self.deconv3d_1,
            self.deconv3d_2,
            self.deconv3d_3
        ]
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.conv = nn.Conv3d(16, 1, 3, padding=(1, 1, 1))

    def forward(self, xs):
        x = None
        for deconv in self.deconv2ds:
            if x is None:
                x = xs.pop()
            else:
                x = torch.cat((x, xs.pop()), dim=1)
            x = deconv(x)
            x = self.relu(x)

        for deconv in self.deconv3ds:
            x = torch.cat((x, xs.pop()), dim=1)
            x = deconv(x)
            x = self.relu(x)

        x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class EncoderAndDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)