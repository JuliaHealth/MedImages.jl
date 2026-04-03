import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
class Feature_extractor(nn.Module):
  def __init__(self, in_channels, out_channels, num_filters): # 2 8 4
    super(Feature_extractor, self).__init__()
    self.model_3D = nn.Sequential(nn.Conv3d(in_channels, num_filters, kernel_size= (7,7,5), stride= (1,1,1), padding=(3,3,1), bias = False),
                                  nn.BatchNorm3d(num_filters),
                                  nn.LeakyReLU(0.1, True),
                                  nn.MaxPool3d(kernel_size= (1,1,2), ceil_mode= True),
                                  nn.Conv3d(num_filters, out_channels, kernel_size= (7,7,3), stride= (1,1,1), padding=(3,3,0), bias = False),
                                  nn.BatchNorm3d(out_channels),
                                  nn.LeakyReLU(0.1, True),
                                  nn.Conv3d(out_channels, out_channels, kernel_size= (7,7,3), stride= (1,1,1), padding=(3,3,0), bias = True),
                              )
  def forward(self, x):
      x = self.model_3D(x)
      x = torch.squeeze(x, dim = -1)
      return x


class Down_Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, mode = 'double_conv'):
        super().__init__()
        assert mode in {'double_conv', 'single_conv'}
        if not mid_channels:
            mid_channels = out_channels
        if mode == 'double_conv':
          self.conv = nn.Sequential(
              nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias = False),
              nn.BatchNorm2d(mid_channels),
              nn.LeakyReLU(0.1, inplace=True),
              nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias = False),
              nn.BatchNorm2d(out_channels),
              nn.LeakyReLU(0.1, inplace=True)
          )
        else:
          self.conv = nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False),
              nn.BatchNorm2d(mid_channels),
              nn.LeakyReLU(0.1, inplace=True)
          )


    def forward(self, x):
        return self.conv(x)

class Up_Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, mode = 'double_conv'):
        super().__init__()
        assert mode in {'double_conv', 'single_conv'}
        if not mid_channels:
            mid_channels = out_channels
        if mode == 'double_conv':
          self.conv = nn.Sequential(
              nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias = False),
              nn.BatchNorm2d(mid_channels),
              nn.ReLU(inplace=True),
              nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias = False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)
          )
        else:
          self.conv = nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)
          )


    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mode = 'double_conv'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Down_Conv(in_channels, out_channels, mode = mode)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, nearest=True, mode = 'double_conv'):
        super().__init__()

        # if nearest, use the normal convolutions to reduce the number of channels
        if nearest:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv = Up_Conv(in_channels, out_channels // 2, in_channels // 2, mode = mode)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = Up_Conv(in_channels, out_channels, mode = mode)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        # print('sizes',x1.size(),x2.size(),diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, init_filter=64, num_down=4, nearest=True, is_single=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.nearest = nearest
        self.num_down = num_down
        factor = 2 if nearest else 1
        filters = []
        for i in range(num_down + 1):
            filters.append(init_filter * (2 ** i))
        # print(filters)
        self.inc = Down_Conv(n_channels, filters[0])
        down_list = []
        up_list = []
        if is_single:
            for i in range(num_down):
                if i == num_down - 1:
                    down_list.append(Down(filters[i], filters[i + 1] // factor, mode='single_conv'))
                else:
                    down_list.append(Down(filters[i], filters[i + 1], mode='single_conv'))
            self.down = nn.ModuleList(down_list)
            for i in range(num_down):
                if i == num_down - 1:
                    up_list.append(
                        Up(filters[num_down - i], filters[num_down - i - 1] * factor, nearest, mode='double_conv'))
                else:
                    up_list.append(Up(filters[num_down - i], filters[num_down - i - 1], nearest, mode='single_conv'))
                self.up = nn.ModuleList(up_list)
        else:
            for i in range(num_down):
                if i == num_down - 1:
                    down_list.append(Down(filters[i], filters[i + 1] // factor, mode='double_conv'))
                else:
                    down_list.append(Down(filters[i], filters[i + 1], mode='double_conv'))
            self.down = nn.ModuleList(down_list)
            for i in range(num_down):
                if i == num_down - 1:
                    up_list.append(
                        Up(filters[num_down - i], filters[num_down - i - 1] * factor, nearest, mode='double_conv'))
                else:
                    up_list.append(Up(filters[num_down - i], filters[num_down - i - 1], nearest, mode='double_conv'))
                self.up = nn.ModuleList(up_list)

        self.outc = OutConv(filters[0], n_classes)

    def forward(self, x):
        down = []
        x = self.inc(x)
        down.append(x)
        for i, down_module in enumerate(self.down):
            x = down_module(x)
            down.append(x)
        for i, up_module in enumerate(self.up):
            if i == 0:
                x = up_module(down[-1], down[-2])
            else:
                x = up_module(x, down[self.num_down - 1 - i])
        logits = self.outc(x)
        return logits
