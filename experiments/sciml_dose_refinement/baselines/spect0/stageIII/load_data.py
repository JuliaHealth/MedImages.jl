from scipy.io import loadmat
import torch
import os
# from glob import glob
import torch.utils.data as data
import numpy as np
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
# torch.manual_seed(0)
# from torchvision import transforms
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class y90loader(data.Dataset):
    def __init__(self, datasetDir='.', mode = 'train', normalize = True):
        super(y90loader, self).__init__()
        assert mode in {'train', 'test'}
        self.pattern = "*.mat"
        self.read_files = []
        self.chunk_depth = 11  # Maybe 7 later
        self.height = 512
        self.width = 512
        self.scale_factor = 1e8
        self.dir = datasetDir
        self.mode = mode
        self.normalize = normalize
        # self.specfic_file = ['petvp01', 'petvp02', 'petvp04', 'petvp05', 'petvp09', 'petvp11', 'petvp12', 'petvp13', 'petvp14']
        self.specfic_file = []
        # self.spect_mean = 0.4082
        # self.spect_std = 1.7851
        # self.density_mean = 258.0605
        # self.density_std = 428.8764
        # self.VDK_mean = 2.6959e-7
        # self.VDK_std = 1.0302e-6
        # self.GT_mean = 2.7890e-7
        # self.GT_std = 1.0555e-6
        if self.normalize:
            print('Input normalization turns on!')
        else:
            print('Input normalization turns off!')
        if self.mode == 'train':
            self.meta = {'spect': [], 'density': [], 'dose_VDK': [], 'dose_GT': [], 'mask': []}
        else:
            self.meta = {'spect': [], 'density': [], 'dose_VDK': []}
        # if self.dir is not None:
        #     for root, dir, files in os.walk(self.dir):
        #         self.read_files.extend(glob(os.path.join(root, self.pattern)))
        # print(len(self.read_files))
        # if len(self.read_files) != 0:
        self.convert2tensor()
        self.meta['spect'] = torch.stack(self.meta['spect']).reshape(-1, 1, self.height, self.width, self.chunk_depth)
        self.meta['density'] = torch.stack(self.meta['density']).reshape(-1, 1, self.height, self.width,
                                                                         self.chunk_depth)
        self.meta['dose_VDK'] = torch.stack(self.meta['dose_VDK']).reshape(-1, 1, self.height, self.width)


        if self.mode == 'train':
            self.meta['mask'] = torch.stack(self.meta['mask']).reshape(-1, 1, self.height, self.width)
            self.meta['dose_GT'] = torch.stack(self.meta['dose_GT']).reshape(-1, 1, self.height, self.width)
        print('spect shape: ', self.meta['spect'].shape)
        print('density shape: ', self.meta['density'].shape)
        print('dose_VDK shape: ', self.meta['dose_VDK'].shape)
        if self.mode == 'train':
            print('mask shape: ', self.meta['mask'].shape)
            print('dose_GT shape: ', self.meta['dose_GT'].shape)
        print('spect max', torch.max(self.meta['spect']).item())
        print('density max', torch.max(self.meta['density']).item())
        print('dose_VDK max', torch.max(self.meta['dose_VDK']).item())
        print('spect mean', torch.mean(self.meta['spect']).item())
        print('density mean', torch.mean(self.meta['density']).item())
        print('dose_VDK mean', torch.mean(self.meta['dose_VDK']).item())
        if self.mode == 'train':
            print('dose_GT max', torch.max(self.meta['dose_GT']).item())
            print('dose_GT mean', torch.mean(self.meta['dose_GT']).item())

    def __getitem__(self, index):
        Meta = {}
        Meta['spect'] = self.meta['spect'][index, :, :, :, :]
        Meta['density'] = self.meta['density'][index, :, :, :, :]
        Meta['dose_VDK'] = self.meta['dose_VDK'][index, :, :, :]
        if self.mode == 'train':
            Meta['mask'] = self.meta['mask'][index, :, :, :]
            Meta['dose_GT'] = self.meta['dose_GT'][index, :, :, :]
        return Meta
    def __len__(self):
        return self.meta['spect'].shape[0]
    def convert2tensor(self):
        path_list= []
        file_list = []
        for root, _, files in os.walk(self.dir):
            path_list.append(root)
            files = [fi for fi in files if fi.endswith(".mat")]
            # print(files)
            file_list.append(files)
        file_num = len(path_list) - 1
        pad_size = self.chunk_depth // 2
        m = nn.ReplicationPad3d((pad_size, pad_size, 0, 0, 0, 0))
        spect_str = 'spect'
        density_str = 'denmap'
        doseGT_str = 'doseGT'
        doseVDK_str = 'doseDVK'
        mask_str = 'mask'
        for i in range(file_num):
            matfiles = file_list[i + 1]
            if self.specfic_file != []:
                if not any(specfic in path_list[i + 1] for specfic in self.specfic_file):
                    print('{} phantom is excluded!'.format(path_list[i + 1]))
                    continue
            if len(matfiles) < 3:
                raise FileNotFoundError('Mat files not found!')
            print('load data from {}'.format(path_list[i + 1]))
            for s in matfiles:
                if spect_str in s:
                    spect_matching = s
                    print('load spect map from {}'.format(os.path.join(path_list[i + 1], spect_matching)))
                    spect = torch.tensor(loadmat(os.path.join(path_list[i + 1], spect_matching))['spect'], # zongyu: ['x']
                                         dtype=torch.float)
                    if self.normalize:
                        print('spect is normalized!')
                        spect = spect / torch.sum(spect) * 0.024 / (0.0976562 * 0.0976562 * 0.2) * self.scale_factor
                        # spect = (spect - self.spect_mean) / self.spect_std
                    spect = m(spect.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                    spect = spect.unfold(2, self.chunk_depth, 1).permute(2, 0, 1, 3)
                    self.meta['spect'].append(spect)
                if density_str in s:
                    density_matching = s
                    print('load density map from {}'.format(os.path.join(path_list[i + 1], density_matching)))
                    density = torch.tensor(loadmat(os.path.join(path_list[i + 1], density_matching))['denmap'].astype(np.float),
                                         dtype=torch.float)
                    density = m(density.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                    density = density.unfold(2, self.chunk_depth, 1).permute(2, 0, 1, 3)
                    self.meta['density'].append(density)
                if mask_str in s:
                    mask_matching = s
                    if self.mode == 'train':
                        print('load mask map from {}'.format(os.path.join(path_list[i + 1], mask_matching)))
                        mask = torch.tensor(loadmat(os.path.join(path_list[i + 1], mask_matching))['mask'], dtype=torch.float)
                        mask = mask.unfold(2, 1, 1).permute(2, 0, 1, 3).squeeze(-1)
                        self.meta['mask'].append(mask)
                if doseGT_str in s:
                    doseGT_matching = s
                    if self.mode == 'train':
                        print('load doseGT map from {}'.format(os.path.join(path_list[i + 1], doseGT_matching)))
                        doseGT = torch.tensor(loadmat(os.path.join(path_list[i + 1], doseGT_matching))['dosemap_gt'], dtype=torch.float) * self.scale_factor
                        doseGT = doseGT.unfold(2, 1, 1).permute(2, 0, 1, 3).squeeze(-1)
                        self.meta['dose_GT'].append(doseGT)
                if doseVDK_str in s:
                    doseVDK_matching = s
                    print('load doseVDK map from {}'.format(os.path.join(path_list[i + 1], doseVDK_matching)))
                    doseVDK = torch.tensor(loadmat(os.path.join(path_list[i + 1], doseVDK_matching))['dosemap_dvk23'], dtype=torch.float) * self.scale_factor
                    doseVDK = doseVDK.unfold(2, 1, 1).permute(2, 0, 1, 3).squeeze(-1)
                    self.meta['dose_VDK'].append(doseVDK)
            print('data load {} / {} finished!'.format(i + 1, file_num))


class data_loader():
    def __init__(self, dir, mode, batch_size, normalize = True, valid_percent = 0.2):
        assert mode in {'train', 'test'}
        self.mode = mode
        self.dir = dir
        self.batch_size = batch_size
        self.valid_percent = valid_percent
        self.normalize = normalize
    def load(self):
        if self.mode == 'train':
            loader = y90loader(datasetDir=self.dir, mode= self.mode, normalize= self.normalize)
            num_valid = int(len(loader) * self.valid_percent)
            train, val = data.random_split(loader, [len(loader) - num_valid, num_valid])
            train_dataloader = data.DataLoader(train, batch_size= self.batch_size, shuffle= True, num_workers= 8, pin_memory= True)
            val_dataloader = data.DataLoader(val, batch_size= self.batch_size, shuffle= False, num_workers= 8, pin_memory= True)
            print(val_dataloader.dataset.indices)
            return train_dataloader, val_dataloader
        else:
            loader = y90loader(datasetDir=self.dir, mode= self.mode, normalize= self.normalize)
            test_dataloader = data.DataLoader(loader, batch_size= self.batch_size, shuffle= False, num_workers= 8, pin_memory= True)
            return test_dataloader

if __name__ == '__main__':
    train_dir = os.path.join(os.getcwd(), 'train')
    loader = y90loader(datasetDir = train_dir, normalize= True)
