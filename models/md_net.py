import torch.nn.functional as F
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=(1, 1), stride=1)
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        cat_x = torch.cat((x1, x2), 1)
        output = self.conv(cat_x)
        return output


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


'''
model Two_decoder_Net
'''


class Two_decoder_Net(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(Two_decoder_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder
        self.up1_1 = Up(1024, 512, bilinear)
        self.up1_2 = Up(1024, 512, bilinear)
        self.up2_1 = Up(512, 256, bilinear)
        self.up2_2 = Up(512, 256, bilinear)
        self.up3_1 = Up(256, 128, bilinear)
        self.up3_2 = Up(256, 128, bilinear)
        self.up4_1 = Up(128, 64, bilinear)
        self.up4_2 = Up(128, 64, bilinear)
        self.out_1 = OutConv(64, n_classes)
        self.out_2 = OutConv(64, n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        o_4_1 = self.up1_1(x5, x4)
        o_4_2 = self.up1_2(x5, x4)

        o_3_1 = self.up2_1(o_4_1, x3)
        o_3_2 = self.up2_2(o_4_2, x3)

        o_2_1 = self.up3_1(o_3_1, x2)
        o_2_2 = self.up3_2(o_3_2, x2)

        o_1_1 = self.up4_1(o_2_1, x1)
        o_1_2 = self.up4_2(o_2_2, x1)

        o_seg1 = self.out_1(o_1_1)
        o_seg2 = self.out_2(o_1_2)

        if self.n_classes > 1:
            seg1 = F.softmax(o_seg1, dim=1)
            seg2 = F.softmax(o_seg2, dim=1)
            return seg1, seg2
        elif self.n_classes == 1:
            seg1 = torch.sigmoid(o_seg1)
            seg2 = torch.sigmoid(o_seg2)
            return seg1, seg2


'''
model Multi-decoder-Net
'''


class Multi_decoder_Net(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(Multi_decoder_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # decoder
        self.up1_1 = Up(1024, 512, bilinear)
        self.up1_2 = Up(1024, 512, bilinear)
        self.up1_3 = Up(1024, 512, bilinear)

        self.up2_1 = Up(512, 256, bilinear)
        self.up2_2 = Up(512, 256, bilinear)
        self.up2_3 = Up(512, 256, bilinear)

        self.up3_1 = Up(256, 128, bilinear)
        self.up3_2 = Up(256, 128, bilinear)
        self.up3_3 = Up(256, 128, bilinear)

        self.up4_1 = Up(128, 64, bilinear)
        self.up4_2 = Up(128, 64, bilinear)
        self.up4_3 = Up(128, 64, bilinear)

        self.out_1 = OutConv(64, n_classes)
        self.out_2 = OutConv(64, n_classes)
        self.out_3 = OutConv(64, n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        o_4_1 = self.up1_1(x5, x4)
        o_4_2 = self.up1_2(x5, x4)
        o_4_3 = self.up1_3(x5, x4)

        o_3_1 = self.up2_1(o_4_1, x3)
        o_3_2 = self.up2_2(o_4_2, x3)
        o_3_3 = self.up2_3(o_4_3, x3)

        o_2_1 = self.up3_1(o_3_1, x2)
        o_2_2 = self.up3_2(o_3_2, x2)
        o_2_3 = self.up3_3(o_3_3, x2)

        o_1_1 = self.up4_1(o_2_1, x1)
        o_1_2 = self.up4_2(o_2_2, x1)
        o_1_3 = self.up4_3(o_2_3, x1)

        o_seg1 = self.out_1(o_1_1)
        o_seg2 = self.out_2(o_1_2)
        o_seg3 = self.out_3(o_1_3)

        if self.n_classes > 1:
            seg1 = F.softmax(o_seg1, dim=1)
            seg2 = F.softmax(o_seg2, dim=1)
            seg3 = F.softmax(o_seg3, dim=1)
            return seg1, seg2, seg3
        elif self.n_classes == 1:
            seg1 = torch.sigmoid(o_seg1)
            seg2 = torch.sigmoid(o_seg2)
            seg3 = torch.sigmoid(o_seg3)
            return seg1, seg2, seg3


