import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet3d(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, init_filter=16):
        super().__init__()
        self.down1 = ConvBlock3d(in_ch, init_filter)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = ConvBlock3d(init_filter, init_filter * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = ConvBlock3d(init_filter * 2, init_filter * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.bridge = ConvBlock3d(init_filter * 4, init_filter * 8)
        
        self.up3 = nn.ConvTranspose3d(init_filter * 8, init_filter * 4, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock3d(init_filter * 8, init_filter * 4)
        
        self.up2 = nn.ConvTranspose3d(init_filter * 4, init_filter * 2, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock3d(init_filter * 4, init_filter * 2)
        
        self.up1 = nn.ConvTranspose3d(init_filter * 2, init_filter, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock3d(init_filter * 2, init_filter)
        
        self.out_conv = nn.Conv3d(init_filter, out_ch, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        
        b = self.bridge(p3)
        
        u3 = self.up3(b)
        u3 = torch.cat([d3, u3], dim=1)
        u3 = self.up_conv3(u3)
        
        u2 = self.up2(u3)
        u2 = torch.cat([d2, u2], dim=1)
        u2 = self.up_conv2(u2)
        
        u1 = self.up1(u2)
        u1 = torch.cat([d1, u1], dim=1)
        u1 = self.up_conv1(u1)
        
        return self.out_conv(u1)

class Spect0Net(nn.Module):
    """Adapted from spect0: U-Net + Residual baseline"""
    def __init__(self, in_ch=2, out_ch=1):
        super().__init__()
        self.unet = UNet3d(in_ch, out_ch)
    def forward(self, spect, density, approx_dose):
        x = torch.cat([spect, density], dim=1)
        res = self.unet(x)
        return F.softplus(res + approx_dose)

class DblurDoseNet(nn.Module):
    """Adapted from DblurDoseNet: Plain 3D U-Net"""
    def __init__(self, in_ch=2, out_ch=1):
        super().__init__()
        self.unet = UNet3d(in_ch, out_ch)
    def forward(self, spect, density):
        x = torch.cat([spect, density], dim=1)
        return F.softplus(self.unet(x))
