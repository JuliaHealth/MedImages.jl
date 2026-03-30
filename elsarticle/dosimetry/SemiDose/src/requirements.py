import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import numpy as np
import statistics
import pandas as pd
from PIL import Image, ImageOps
import SimpleITK as sitk
from datetime import datetime
import random
# deep learning libraries
import timm
import torchvision
import torchvision.transforms as transforms # data augmentation
import torch
import torch.nn as nn
import torch.nn.functional as F # activation, loss, pool, etc.
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam, AdamW
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, R2Score, PearsonCorrCoef
from itertools import zip_longest