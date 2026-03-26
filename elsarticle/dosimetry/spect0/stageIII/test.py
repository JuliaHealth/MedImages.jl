from model import My_model, My_model_2d
from load_data import data_loader
import torch
import torch.nn as nn
import os
import time
import argparse
from scipy.io import savemat

test_dir = os.path.join(os.getcwd(), 'test')
checkpoint_dir= os.getcwd() + '/checkpoint_res_3d_unet/'
save_dir = os.getcwd() + '/result_res_5b'  # ???
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, help='the test batch size', default=1)
parser.add_argument('--is_best', action='store_true', help='load best / last checkpoint')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=50093)
args = parser.parse_args()
scale_factor = 1e8

class test_unet():
    def __init__(self,
                 test_dir,
                 checkpoint_dir,
                 batch_size,
                 is_best = False):

        p = data_loader(dir=test_dir, batch_size=batch_size, mode='test', normalize = True)
        # self.train_data, self.val_data = p.get_data(mode = 'Train', valid_percent = valid_percent)
        self.test_data = p.load()
        # print('Saving GT!')
        # torch.save(self.test_data.dataset, os.getcwd() + '/result/ground_truth.pt')
        self.model = My_model(in_channels=2,
                            out_channels=1,
                            mid_channels=8,
                            mode= 'test')
        # self.model = My_model_2d(in_channels=22,
        #                          out_channels=1,
        #                          mode='test')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if is_best:
            checkpoint_path = checkpoint_dir + 'best_checkpoint.pytorch'
            print('Now loading the best checkpoint!')
        else:
            checkpoint_path = checkpoint_dir + 'last_checkpoint.pytorch'
            print('Now loading the last checkpoint!')
        try:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location= 'cpu'))
            print('load pre-trained model successfully from: {}!'.format(checkpoint_path))
        except:
            raise IOError(f"load Checkpoint '{checkpoint_path}' failed! ")

        '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print("There are", torch.cuda.device_count(), "GPUs!")
                devices = [0, 1]
                self.model = nn.DataParallel(self.model, device_ids=devices)
                print("But Let's use", len(devices), "GPUs!")
        self.model.to(self.device)
        '''

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)


    def test(self):

        with torch.no_grad():
            results = []
            self.model.eval()
            test_loader = self.test_data
            image_num = len(test_loader.dataset)
            for iter_num, batch in enumerate(self.test_data):

                for key in batch:
                    batch[key] = batch[key].to(self.device)

                preds = self.model(batch['spect'], batch['density'], batch['dose_VDK'])

                results.append(preds.cpu())

                print('(test {} / {})'.format(iter_num + 1, image_num // batch['spect'].shape[0]))
            return results

def main():
    start_t = time.time()
    Unet = test_unet(test_dir= test_dir,
                     checkpoint_dir= checkpoint_dir,
                     batch_size= args.batch,
                     is_best= args.is_best)
    results = Unet.test()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # print('Saving Network Output!')
    # torch.save(results, save_dir + '/network_output.pt')
    print('Saving Network Output into .mat file!')
    pred = torch.stack(results[0:-1]).reshape(-1, 512, 512)
    pred = torch.cat((pred, results[-1]), dim= 0) / scale_factor
    savemat(save_dir + '/network_output.mat', {'pred': pred.permute(1, 2, 0).numpy()}, do_compression= True)
    end_t = time.time()
    print('time elapsed is: {:.1f}s'.format(end_t - start_t))
if __name__ == '__main__':
    main()
