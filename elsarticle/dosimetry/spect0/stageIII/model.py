import torch.nn as nn
from buildingblocks import Feature_extractor, UNet
import torch
from torch.nn import init


def init_weights(net, init_type='normal', mean = 0.0, init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        # print(classname)
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, mean, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, mean)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, mean)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class My_model(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, mode = 'train'): ## 2 1 8
        super(My_model, self).__init__()
        assert mode in {'train', 'test'}
        self.mode = mode
        self.feature = Feature_extractor(in_channels, mid_channels, mid_channels // 2)
        self.UNet = UNet(mid_channels, out_channels, init_filter=16, num_down=4, nearest= True, is_single= False)
        # self.GT_mean = 2.7890e-7
        # self.GT_std = 1.0555e-6
        # self.density_mean = 258.0605
        # self.density_std = 428.8764
        init_weights(self.feature, init_type= 'kaiming', mean= 0, init_gain= 0.02)
        init_weights(self.UNet, init_type= 'kaiming', mean= 0, init_gain= 0.02)

    def forward(self, spect, density, quick_dose, visualization = False):
        middle = density.shape[-1] // 2
        middle_density = density[:, :, :, :, middle]
        # VDK = quick_dose.unsqueeze(-1).repeat(1, 1, 1, 1, 11)
        input = torch.cat((spect, density), dim= 1)
        feature_input = self.feature(input)
        if visualization:
            return feature_input
        else:
            out = self.UNet(feature_input)
            out = out + quick_dose
            # print(out.shape)
            if self.mode == 'test':
                # out = out * self.GT_std + self.GT_mean
                out = torch.clamp(out, min = 0)
                out[(middle_density) <= 100] = 0
                out = out.squeeze(1)
            return out


class My_model_2d(nn.Module):
    def __init__(self, in_channels, out_channels, mode='train'):
        super(My_model_2d, self).__init__()
        assert mode in {'train', 'test'}
        self.mode = mode
        self.UNet = UNet(in_channels, out_channels, init_filter=16, num_down=4, nearest=True, is_single=False)
        # self.GT_mean = 2.7890e-7
        # self.GT_std = 1.0555e-6
        # self.density_mean = 258.0605
        # self.density_std = 428.8764
        init_weights(self.UNet, init_type='kaiming', mean=0, init_gain=0.02)

    def forward(self, spect, density, quick_dose):
        # spect is in 3d, B * C * H * W * D
        B, C, H, W, D = spect.shape
        middle = density.shape[-1] // 2
        middle_density = density[:, :, :, :, middle]
        spect = spect.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W).contiguous()
        density = density.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W).contiguous()
        # VDK = quick_dose.unsqueeze(-1).repeat(1, 1, 1, 1, 11)
        input = torch.cat((spect, density), dim=1)
        out = self.UNet(input)
        out = out + quick_dose
        # print(out.shape)
        if self.mode == 'test':
                # out = out * self.GT_std + self.GT_mean
            out = torch.clamp(out, min=0)
            out[(middle_density) <= 100] = 0
            out = out.squeeze(1)
        return out


if __name__ == '__main__':
    test_model = My_model(in_channels=2,
                            out_channels=1,
                            mid_channels= 16,
                            mode= 'train')
    pytorch_total_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)

    print('total params: ', pytorch_total_params)