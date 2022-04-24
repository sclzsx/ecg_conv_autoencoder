import os
import json
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# from argparse import ArgumentParser

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import AutoencoderDataset

# from complex.dataset import AutoencoderDataset, get_labels_weight
# from choices import choose_model, choose_loss

# from sklearn.metrics import f1_score

# from test import autoencoder_test, classifier_test, cascaded_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择用cpu还是gpu


class autoencoder(nn.Module):
    def __init__(self, dim=256):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, dim), nn.Sigmoid())

        self.dim = dim

    def forward(self, x):
        # print(x.shape)

        x = self.encoder(x)
        # print(x.shape)

        x = self.decoder(x)
        # print(x.shape)
        return x

def train():
    max_epoch = 50
    save_dir = 'results/autoencoder/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_dataset = AutoencoderDataset('./data/normal_train.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss(reduction='mean')

    for epoch in range(1, max_epoch + 1):  # range的区间是左闭右开,所以加1

        model.train()  # 训练模式

        mean_loss = 0

        for i, batch_data in enumerate(train_dataloader):  # 遍历train_dataloader,每次返回一个批次,i记录批次id
            
            batch_data = batch_data.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式

            optimizer.zero_grad()  # 清零已有梯度

            batch_pred = model(batch_data)  # 前向传播,获得网络的输出

            batch_loss = criterion(batch_pred, batch_data)

            batch_loss_val = batch_loss.item()

            if i % 100 == 0:
                print(epoch, i, batch_loss_val)

            mean_loss = mean_loss + batch_loss_val  # 累加所有批次的平均损失. item()的意思是取数值,因为该变量不是一个tensor

            batch_loss.backward()  # 反向传播损失,更新模型参数

            optimizer.step()  # 更新学习率

        mean_loss = mean_loss / (i + 1)  # 损失和求均,为当前epoch的损失

        log = {'epoch': epoch, 'loss': mean_loss}
        print(log)

        torch.save(model.state_dict(), save_dir + str(epoch) + '.pth')


if __name__ == '__main__':

    train()
