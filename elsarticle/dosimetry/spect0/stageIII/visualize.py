from model import My_model
from load_data import data_loader
import torch
import torch.nn as nn
import os
import argparse
from scipy.io import savemat

test_dir = os.getcwd() + '/test'
checkpoint_dir= os.getcwd() + '/checkpoint/'
save_dir = os.getcwd() + '/result'
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, help='the test batch size', default=1)
parser.add_argument('--idx', type=int, help='index to visualize', default=70)
parser.add_argument('--is_double', action='store_true', help='convert datatype to double')
parser.add_argument('--is_best', action='store_true', help='load best / last checkpoint')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=50093)
args = parser.parse_args()

class test_unet():
    def __init__(self,
                 test_dir,
                 is_double,
                 checkpoint_dir,
                 batch_size,
                 is_best = False,
                 idx = 70):
        p = data_loader(dir=test_dir, batch_size=batch_size, mode='test')
        # slice from 1 to 130
        self.batch_idx = (idx -1) // batch_size
        self.idx = (idx - 1) % batch_size
        self.test_data = p.load()
        # print('Saving GT!')
        # torch.save(self.test_data.dataset, os.getcwd() + '/result/ground_truth.pt')
        self.model = My_model(in_channels=2,
                              out_channels=1,
                              mid_channels=16,
                              mode='test')
        if is_best:
            checkpoint_path = checkpoint_dir + 'best_checkpoint.pytorch'
        else:
            checkpoint_path = checkpoint_dir + 'last_checkpoint.pytorch'
        try:
            self.model.load_state_dict(torch.load(checkpoint_path))
            print('load pre-trained model successful!')
        except:
            raise IOError(f"load Checkpoint '{checkpoint_path}' failed! ")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print("There are", torch.cuda.device_count(), "GPUs!")
                print("But Let's use the first two GPUs!")
                self.model = nn.DataParallel(self.model, device_ids=[0, 1])
        self.model.to(self.device)
        self.is_double = is_double
        if self.is_double:
            self.model = self.model.double()
    def test(self):
        with torch.no_grad():
            results = []
            self.model.eval()
            test_loader = self.test_data
            image_num = len(test_loader.dataset)
            for iter_num, batch in enumerate(self.test_data):
                if iter_num == self.batch_idx + 1:
                    break
                '''
                batch_data = batch[:, 0:2, :, :, :]
                # batch_gt = batch[:,2:,:,:]
                batch_data = batch_data.to(self.device)
                # batch_gt = batch_gt.to(self.device)
                if self.is_double:
                    batch_data = batch_data.double()
                    # batch_gt = batch_gt.double()
                batch_spect = batch_data[:, 0:1, :, :, :]
                batch_ct = batch_data[:, 1:, :, :, :]
                preds = self.model.forward(batch_spect, batch_ct,  visualization = True)
                '''
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                    if self.is_double:
                        batch[key] = batch[key].double()
                preds = self.model.forward(batch['spect'], batch['density'], batch['dose_VDK'])
                # print(preds.shape)
                results.append(preds.cpu())
                print('(test {} / {})'.format(iter_num + 1, image_num // batch.shape[0]))

            return results[self.batch_idx][self.idx, :, :, :]

def main():

    Unet = test_unet(test_dir= test_dir,
                     is_double= args.is_double,
                     checkpoint_dir= checkpoint_dir,
                     batch_size= args.batch,
                     is_best= args.is_best,
                     idx = args.idx)
    results = Unet.test()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # print('Saving Network Output!')
    # torch.save(results, save_dir + '/network_output.pt')
    print('Saving Network Output into .mat file!')
    pred = results.reshape(-1, 512, 512)
    savemat(save_dir + '/feature_output.mat', {'pred': pred.permute(1, 2, 0).numpy()}, do_compression= True)


if __name__ == '__main__':
    main()