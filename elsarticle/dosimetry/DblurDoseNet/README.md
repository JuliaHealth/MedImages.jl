# DblurDoseNet

https://github.com/ZongyuLi-umich/DblurDoseNet

PyTorch implementation of DblurDoseNet
method for SPECT dosimetry,
as described in the paper
"DblurDoseNet: A deep residual learning network for voxel radionuclide dosimetry compensating for SPECT imaging resolution",
by
Zongyu Li, Jeffrey A Fessler, Justin K Mikell, Scott J Wilderman, Yuni K Dewaraja;
Medical Physics 49(2):1216-30, Feb. 2021.
https://doi.org/10.1002/mp.15397

# Dataset 

The dataset used for training and testing is available here: https://doi.org/10.7302/ykz6-cn05

# Data Configuration

To replicate the paper, you must structure the downloaded dataset correctly. The training script expects a specific folder layout and particular files as input.
## 1. Directory Structure

After downloading and unzipping the dataset, you should organize the folders as follows. 
Create a main data directory (e.g., `DblurDoseNet_data`) and place the `train`, `test`, and `DVKs` folders inside it.

**The training data should follow this pattern:**
- DblurDoseNet_data/train/petvp01/spect.mat
- DblurDoseNet_data/train/petvp01/density.mat
- DblurDoseNet_data/train/petvp01/doseGT.mat
... and so on for all files in petvp01.

- DblurDoseNet_data/train/petvp04/...
...and so on for all other training phantoms.

**The testing data should follow the same pattern:**
- DblurDoseNet_data/test/petvp6/spect.mat
- DblurDoseNet_data/test/petvp6/density.mat
- DblurDoseNet_data/test/petvp6/doseGT.mat
...and so on for all files in petvp6.

- DblurDoseNet_data/test/petvp7/...
...and so on for all other testing phantoms.

**The DVK kernels should be placed in their own folder:**
- DblurDoseNet_data/DVKs/beta.npy
- DblurDoseNet_data/DVKs/gamma.npy

## 2. Model Input and Output

The DblurDoseNet model is designed to learn a mapping from the low-resolution SPECT image to the high-fidelity ground-truth dose-rate map, using the density map as additional information. For each phantom:

**Input: The model takes two .mat files as input:**
- spect.mat: The SPECT reconstruction image.
- density.mat: The density map.

**Target (Ground Truth):** The model learns to produce the doseGT.mat file, which is the ground-truth dose-rate map.
The train.py script will load these files from the train directory during the training process.

# Training
To train the model, you need to point the training script to your dataset directory. The command below includes a `--data_path` argument for this purpose (Note: the exact argument name may vary; 
please check the train.py script if needed).
```python
python3 train.py --data_path [/path/to/DblurDoseNet_data] --batch [batch size] --lr [learning rate] --epochs [# of epochs] <br>
```
For example, with a batch size of 32, a learning rate of 0.002, and 200 epochs, the training command is: <br> 
```python
python3 train.py --data_path ./DblurDoseNet_data --batch 32 --lr 0.002 --epochs 200
```

# Testing
To use the best checkpoint from training (the one with the lowest validation loss), run the following command. Remember to point to your test data as well.
```python
python3 test.py --data_path [/path/to/DblurDoseNet_data] --batch [batch size] --is_best
```
