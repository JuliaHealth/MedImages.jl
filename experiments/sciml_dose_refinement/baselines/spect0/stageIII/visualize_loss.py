import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
path = '/Users/lizongyu/Desktop/checkpoint'
train_loss = torch.load('/Users/lizongyu/Desktop/checkpoint/train_loss.pt')
valid_loss = torch.load('/Users/lizongyu/Desktop/checkpoint/valid_loss.pt')
for i in range(len(train_loss)):
    train_loss[i] = train_loss[i] / 1000
    valid_loss[i] = valid_loss[i] / 1000

iter = torch.arange(1,101)
# print(iter)
plt.figure()
plt.plot(iter,train_loss, label='train loss')
plt.plot(iter,valid_loss, label = 'valid loss')
plt.xlabel('num epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig("train_val_loss.pdf", bbox_inches='tight')
plt.show()
