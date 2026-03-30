from UnetTrainer import Unet3DTrainer
from load_data import data_loader
from model import My_model, My_model_2d
import torch
import torch.nn as nn
import argparse
import os
# torch.manual_seed(0)
train_dir = os.path.join(os.getcwd(), '../../data/training_phantom/')
# print(train_dir)
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, help='the train batch size', default=1)
parser.add_argument('--lr', type=float, help='the learning rate', default=2e-3)
parser.add_argument('--epochs', type=int, help='training epochs', default=50)
parser.add_argument('--continue_train', action='store_true', help='load checkpoint and continue training')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=50093)
args = parser.parse_args()
class train_unet():
    def __init__(self,
                 train_dir,
                 batch_size,
                 learning_rate,
                 num_epochs,
                 continue_train,
                 valid_percent):
        p = data_loader(dir=train_dir, batch_size = batch_size, mode='train', normalize = True, valid_percent=valid_percent)
        # self.train_data, self.val_data = p.get_data(mode = 'Train', valid_percent = valid_percent)
        self.train_data, self.val_data = p.load()
        if not os.path.exists(os.getcwd() + '/result_res_2d_unet_shr'):
            os.mkdir(os.getcwd() + '/result_res_2d_unet_shr')
        if not os.path.exists(os.getcwd() + '/checkpoint_res_2d_unet_shr'):
            os.mkdir(os.getcwd() + '/checkpoint_res_2d_unet_shr')
        torch.save(self.val_data.dataset.indices, os.getcwd() + '/result_res_2d_unet_shr/val_ind.pt')
        # self.model = My_model(in_channels=2,
        #                     out_channels=1,
        #                     mid_channels= 8,
        #                     mode= 'train')
        self.model = My_model_2d(in_channels = 22,
                                 out_channels = 1,
                                 mode = 'train')
        # print(self.model)
        checkpoint_dir = os.getcwd() + '/checkpoint_res_2d_unet_shr/'
        checkpoint_path = checkpoint_dir + 'last_checkpoint.pytorch'
        if continue_train:
            try:
                self.model.load_state_dict(torch.load(checkpoint_path))
                print('load pre-trained model successful, continue training!')
            except:
                raise IOError(f"Checkpoint '{checkpoint_path}' load failed! ")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                print("There are", torch.cuda.device_count(), "GPUs!")
                devices = [0, 1]
                self.model = nn.DataParallel(self.model, device_ids= devices)
                print("But Let's use", len(devices), "GPUs!")
        self.model.to(self.device)
        self.trainer = Unet3DTrainer(model= self.model,
                                     learning_rate= learning_rate,
                                     num_epochs= num_epochs,
                                     batch_size = batch_size,
                                     )
    def train(self):
        self.trainer.train(self.train_data, self.val_data)
def main():

    Unet = train_unet(train_dir= train_dir,
                      batch_size= args.batch,
                      learning_rate= args.lr,
                      num_epochs= args.epochs,
                      continue_train= args.continue_train,
                      valid_percent= 0.2)
    Unet.train()

if __name__ == '__main__':
    main()