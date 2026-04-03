import os
import torch
import time
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import save_checkpoint
# from ssim import SSIM
import torch.nn as nn
# import nvidia_smi

# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
'''
def _check_gpu():
    global _GPU
    global _NUMBER_OF_GPU
    nvidia_smi.nvmlInit()
    _NUMBER_OF_GPU = nvidia_smi.nvmlDeviceGetCount()
    if _NUMBER_OF_GPU > 0:
        _GPU = True

def _print_gpu_usage(detailed=False):
    if not detailed:
        for i in range(_NUMBER_OF_GPU):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(f'GPU-{i}: GPU-Memory: {_bytes_to_megabytes(info.used)}/{_bytes_to_megabytes(info.total)} MB')



def _bytes_to_megabytes(bytes):
    return round((bytes/1024)/1024,2)
'''


class Unet3DTrainer:
    def __init__(self,
                 model,
                 learning_rate = 2e-3,
                 num_epochs = 20,
                 batch_size = 8
                 ):
        self.model = model

        print('total params: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(
                                    filter(lambda p: p.requires_grad, self.model.parameters()),
                                    learning_rate)  # leave betas and eps by default
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, verbose=True, factor= 0.2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion_l2 = nn.MSELoss(reduction= 'sum')
        self.criterion_l1 = nn.L1Loss(reduction= 'sum')
        # self.ssim = SSIM()
        self.batch_size = batch_size
        self.dir_str_result_3d = 'result_res_3d_unet'
        self.dir_str_checkpoint_3d = 'checkpoint_res_3d_unet'
        # _GPU = False
        # _NUMBER_OF_GPU = 0
        # _check_gpu()

    def my_criterion(self, pred, target, density_mask):
        pred_body = torch.mul(pred, density_mask)
        target_body = torch.mul(target, density_mask)
        pred_outbody = torch.mul(pred, 1 - density_mask)
        target_outbody = torch.mul(target, 1 - density_mask)
        if torch.sum(density_mask) != 0 and torch.sum(1 - density_mask) != 0:
            loss = 0.7 * self.criterion_l2(pred_body, target_body) / torch.sum(density_mask) + \
                0.3 * self.criterion_l2(pred_outbody, target_outbody) / torch.sum(1 - density_mask)
            # ssim_out = - self.ssim(pred_body, target_body)
            # print('SSIM loss: ', ssim_out.item())
            # loss = loss + 10 * ssim_out
        else:
            loss = self.criterion_l2(pred, target)
            # ssim_out = - self.ssim(pred, target)
            # loss = loss + 10 * ssim_out
        # ssim_loss = 1e4 * self.ssim(pred, target)
        # print('l2 loss: ', l2_loss.item())
        # ssim_loss = - 0.7 * self.ssim(pred_body, target_body) - 0.3 * self.ssim(pred_outbody, target_outbody)
        # ssim_loss = ssim_loss * 100
        # print('SSIM loss: ', ssim_loss.item())
        # loss = loss + ssim_loss
        return loss

    def train(self, train_loader, valid_loader):
        image_num = len(train_loader.dataset.indices)
        niter = np.int(np.ceil(image_num / self.batch_size))
        Loss_history = []
        valid_loss_history = []
        Initial_loss = self.valid(valid_loader= valid_loader)
        print('Initial loss: ', Initial_loss)
        for i in range(self.num_epochs):
            self.model.train()
            start_t = time.time()
            acc_loss = 0
            for iter_num, batch in enumerate(train_loader):
            # for iter_num in range(1):
                self.optimizer.zero_grad()
                for key in batch:
                    # print('key: {}, maximum value: {}'.format(key, torch.max(batch[key]).item()))
                    batch[key] = batch[key].to(self.device)
                scores = self.model(batch['spect'], batch['density'], batch['dose_VDK'])
                # loss = self.my_criterion(scores, batch['gt_dose'], batch['density'])
                loss = self.my_criterion(scores, batch['dose_GT'], batch['mask']) # + 100 * self.criterion_3(scores, batch['gt_dose'])
                acc_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                print('(Iter {} / {})'.format(iter_num + 1, niter))
                # _print_gpu_usage()
                # self.writer.add_scalar('Loss/train', loss.item(), i * (image_num // batch.shape[0]) + iter_num)
            end_t = time.time()
            acc_loss = acc_loss / niter
            Loss_history.append(acc_loss)
            valid_loss = self.valid(valid_loader= valid_loader)
            valid_loss_history.append(valid_loss)
            print('(Epoch {} / {}) train loss: {:.4f} valid loss: {:.4f} time per epoch: {:.1f}s current lr: {}'.format(
                i + 1, self.num_epochs, acc_loss, valid_loss, end_t - start_t, self.optimizer.param_groups[0]['lr']))
                       # save_checkpoint(self.model.module.state_dict(), is_best=True, checkpoint_dir=os.getcwd() + '/checkpoint/')
            if (i + 1) % 5 == 0:
                print('Save the current model to checkpoint!')
                save_checkpoint(self.model.module.state_dict(), is_best = False, checkpoint_dir = os.path.join(os.getcwd(), self.dir_str_checkpoint_3d))
                torch.save(Loss_history, os.path.join(os.getcwd(), self.dir_str_result_3d, 'train_loss.pt'))
                torch.save(valid_loss_history, os.path.join(os.getcwd(), self.dir_str_checkpoint_3d, 'valid_loss.pt'))
            if i == np.argmin(valid_loss_history):
                print('The current model is the best model! Save it!')
                save_checkpoint(self.model.module.state_dict(), is_best=True,
                                checkpoint_dir = os.path.join(os.getcwd(), self.dir_str_checkpoint_3d))
            self.lr_scheduler.step(valid_loss)

    def valid(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            image_num = len(valid_loader.dataset.indices)
            niter = np.int(np.ceil(image_num / self.batch_size))
            acc_loss = 0
            for iter_num, batch in enumerate(valid_loader):
                for key in batch:
                    batch[key] = batch[key].to(self.device)

                scores = self.model(batch['spect'], batch['density'], batch['dose_VDK'])
                loss = self.my_criterion(scores, batch['dose_GT'], batch['mask'])  # + 100 * self.criterion_3(scores, batch['gt_dose'])
                # loss = self.criterion_1(scores, batch['gt_dose']) * 1000
                acc_loss += loss.item()
                print('(valid {} / {})'.format(iter_num + 1, niter))

        return acc_loss / niter