import torch
import torch.nn as nn

# model from O. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015
class Conv(nn.Module):
    """ module: Convolution -> Bath Normalization -> ReLU """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool after two conv module"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            Conv(in_ch, out_ch),
            Conv(out_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling after two conv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose1d(in_ch , in_ch // 2, kernel_size=2, stride=2)
        self.conv1 = Conv(in_ch, out_ch)
        self.conv2 = Conv(out_ch, out_ch)

    def forward(self, x1, x2):
        # x1 output of stream/ x2 copy and crop
        x1 = self.up(x1)
        # input is Channel x Height x Width
        diffY = x2.size()[2] - x1.size()[2]

        x2 = x2[:, :, 0:(x2.size()[2] - diffY)]

        x = torch.cat([x2, x1], dim=1) # concat output of stream and cropped features
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_ch, n_class):
        super(UNet, self).__init__()

        self.in1 = Conv(n_ch, 64)
        self.in2 = Conv(64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.out = OutConv(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.FC = nn.Linear(2992, n_class)
    def forward(self, x):
        x = self.in1(x)
        x1 = self.in2(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        x = self.dropout(x)
        return x

    def test(self, x):
        x = self.in1(x)
        x1 = self.in2(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        x = x.view(x.shape[0], -1)
        x = self.FC(x)
        return x

class UNet_spectrogram(nn.Module):
    def __init__(self, n_ch, n_class):
        super(UNet_spectrogram, self).__init__()

        self.in1 = Conv(n_ch, 64)
        self.in2 = Conv(64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.out = OutConv(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.FC = nn.Linear(2992+32*80, n_class)
    def forward(self, x, y):
        x = self.in1(x)
        x1 = self.in2(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        y = y.view(y.shape[0], -1)
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, y), 1)
        x = self.FC(x)
        x = self.dropout(x)
        return x

    def test(self, x, y):
        x = self.in1(x)
        x1 = self.in2(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        y = y.view(y.shape[0], -1)
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, y), 1)
        x = self.FC(x)
        return x


