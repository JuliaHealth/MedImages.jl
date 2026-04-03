from requirements import *
from hyper_parameter import *

img_shape = (3, img_height, img_width)

class Generator_ft(nn.Module): # for feature matching
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(96, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh() # (-1,1)
            #nn.Sigmoid() # (0,1)
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator_ft(nn.Module): # for feature matching
    def __init__(self, model_name='resnet18', num_classes=1, pretrained=True):
        super(Discriminator, self).__init__()
        self.pretrained_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # num_classes=0means remove the last classifer layer
        self.features_dim = self.pretrained_model.num_features
        
        # add customized layers
        self.fc1 = nn.Linear(self.features_dim, 1)  # for real or fake
        self.fc2 = nn.Linear(self.features_dim, num_classes)  # for regression
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pretrained_model(x)  
        x = x.view(x.size(0), -1)  # flatten the feature maps
        x = self.dropout(x)  # Apply dropout
        features = x  # feature layer
        real_fake = torch.sigmoid(self.fc1(x))  
        regression_output = self.fc2(x)  
        return real_fake, regression_output, features


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(256, 256, normalize=False),
            #*block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh() 
        )
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=1, pretrained=True):
        super(Discriminator, self).__init__()
        self.pretrained_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # num_classes=0means remove the last classifer layer
        self.features_dim = self.pretrained_model.num_features
        
        # add customized layers
        self.fc1 = nn.Linear(self.features_dim, 1)  # for real or fake
        self.fc2 = nn.Linear(self.features_dim, num_classes)  # for regression
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pretrained_model(x)  
        x = x.view(x.size(0), -1)  # flatten the feature maps
        x = self.dropout(x)  # Apply dropout
        real_fake = torch.sigmoid(self.fc1(x))  
        regression_output = self.fc2(x)  
        return real_fake, regression_output




"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""

# for mean teacher
consistency = 5
consistency_rampup = 200.0

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

def update_ema_variables(model, ema_model, beta, global_step):
    # Use the true average until the exponential average is more correct
    beta = min(1 - 1 / (global_step + 1), beta)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(beta).add_(param.data, alpha = 1 - beta)




# resnet50 for SimRegMatch
class ResNet(nn.Module):
    def __init__(self, dropout=0.1):
        super(ResNet, self).__init__() 
        self.pretrained_model = timm.create_model('resnet50', pretrained=False, num_classes=0)  # num_classes=0means remove the last classifer layer
        #self.pretrained_model = timm.create_model('caformer_s36', pretrained=True, num_classes=0)  # num_classes=0means remove the last classifer layer
        self.features_dim = self.pretrained_model.num_features

        # add customized layers
        self.linear = nn.Linear(self.features_dim, 1)
        self.dropout = nn.Dropout(p=dropout)  # Dropout for regularization

    def forward(self, x):
        x = self.pretrained_model(x)  
        x = x.view(x.size(0), -1)
        encoding = x

        x = self.dropout(x)
        x = self.linear(x)
        return x, encoding

def resnet50(dropout=0.1): 
    return ResNet(dropout)
