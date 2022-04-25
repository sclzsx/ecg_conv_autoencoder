import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import AutoencoderDataset
from models import autoencoder, convautoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择用cpu还是gpu

batch_size = 64  # 训练阶段的批次大小
max_epoch = 50  # 训练阶段的最大轮数
do_eval = 1  # 是否在训练阶段做验证，并保存验证指标最佳的模型
model_id = 1  # 0为人工神经网络，1为卷积神经网络


def train():
    # 选择模型
    if model_id == 0:
        model = autoencoder().to(device)
        save_dir = 'results/autoencoder/'
    else:
        model = convautoencoder().to(device)
        save_dir = 'results/convautoencoder/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载数据集
    train_dataset = AutoencoderDataset('./data/normal_train.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if do_eval:
        test_dataset1 = AutoencoderDataset('./data/normal_test.csv')
        test_dataset2 = AutoencoderDataset('./data/abnormal_test.csv')
        max_metirc = -1
    print('Finish loading dataset, begin training.')

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss(reduction='mean')

    for epoch in range(1, max_epoch + 1):  # range的区间是左闭右开,所以加1
        model.train()  # 训练模式
        mean_loss = 0
        for i, batch_data in enumerate(train_dataloader):  # 遍历train_dataloader,每次返回一个批次,i记录批次id
            batch_data = batch_data.to(device)  # 若环境可用gpu,自动将tensor转为cuda格式
            optimizer.zero_grad()  # 清零已有梯度
            batch_pred = model(batch_data)  # 前向传播,获得网络的输出
            batch_loss = criterion(batch_pred, batch_data)
            batch_loss_val = batch_loss.item()
            if i % 500 == 0:
                print('Training epoch:{}, iter:{}, loss:{}'.format(epoch, i, batch_loss_val))
            mean_loss = mean_loss + batch_loss_val
            batch_loss.backward()  # 反向传播损失,更新模型参数
            optimizer.step()  # 更新学习率
        mean_loss = mean_loss / (i + 1)  # 损失和求均,为当前epoch的损失
        log = {'epoch': epoch, 'loss': mean_loss}
        print('# Finish training one epoch.', log)

        # 完成该轮训练后进行验证
        if do_eval:
            model.eval()  # 验证模式
            all_loss = []
            for i, batch_data in enumerate(train_dataloader):
                with torch.no_grad():
                    batch_data = batch_data.to(device)
                    batch_pred = model(batch_data)
                    batch_loss = criterion(batch_pred, batch_data)
                    batch_loss_val = batch_loss.item()
                    all_loss.append(batch_loss_val)
            all_loss = np.array(all_loss)
            min_loss = np.min(all_loss)
            max_loss = np.max(all_loss)
            mean_loss = np.mean(all_loss)
            std_loss = np.std(all_loss)
            threshold = mean_loss + std_loss  # 阈值等于均值加标准差

            # normal: P, abnormal: N

            all_loss = []
            TP, FP = 0, 0
            for i, data in enumerate(test_dataset1):
                with torch.no_grad():
                    data = data.unsqueeze(0).to(device)
                    pred = model(data)
                    loss = criterion(pred, data).cpu().numpy()
                    all_loss.append(loss)
                    # if loss < min_loss or loss > max_loss:  # 判定异常信号的方式一：小于最小值或大于最大值判定为异常
                    if loss > threshold:  # 判定异常信号的方式二：大于阈值判定为异常
                        FP = FP + 1
                    else:
                        TP = TP + 1

            all_loss = []
            TN, FN = 0, 0
            for i, data in enumerate(test_dataset2):
                with torch.no_grad():
                    data = data.unsqueeze(0).to(device)
                    pred = model(data)
                    loss = criterion(pred, data).cpu().numpy()
                    all_loss.append(loss)
                    # if loss < min_loss or loss > max_loss:  # 判定异常信号的方式一：小于最小值或大于最大值判定为异常
                    if loss > threshold:  # 判定异常信号的方式二：大于阈值判定为异常
                        TN = TN + 1
                    else:
                        FN = FN + 1

            p = TP / (TP + FP)
            r = TP / (TP + FN)
            f = 2 * p * r / (p + r)
            a = (TP + TN) / (TP + FP + TN + FN)
            if a > max_metirc:
                max_metirc = a  # 更新目前的最佳指标
                log = {'epoch': epoch, 'p': p, 'r': r, 'f1': f, 'a': a}
                print('Find the best metric', log)
                torch.save(model.state_dict(), save_dir + 'best.pth')
        else:
            torch.save(model.state_dict(), save_dir + 'best.pth')

    torch.save(model.state_dict(), save_dir + 'last.pth')


def cal_threshold():
    if model_id == 0:
        model = autoencoder().to(device)
        save_dir = 'results/autoencoder/'
    else:
        model = convautoencoder().to(device)
        save_dir = 'results/convautoencoder/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_dataset = AutoencoderDataset('./data/normal_train.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('Finish loading dataset, begin calculating threshold.')

    criterion = torch.nn.L1Loss(reduction='mean')

    model_path = save_dir + 'best.pth'
    model.load_state_dict(torch.load(model_path))

    all_loss = []
    for i, batch_data in enumerate(train_dataloader):
        with torch.no_grad():
            batch_data = batch_data.to(device)
            batch_pred = model(batch_data)
            batch_loss = criterion(batch_pred, batch_data)
            batch_loss_val = batch_loss.item()
            all_loss.append(batch_loss_val)
    all_loss = np.array(all_loss)
    min_loss = np.min(all_loss)
    max_loss = np.max(all_loss)
    mean_loss = np.mean(all_loss)
    std_loss = np.std(all_loss)

    thresh_path = save_dir + 'thresh.txt'
    with open(thresh_path, 'w') as f:
        f.write(str(min_loss))
        f.write(' ')
        f.write(str(max_loss))
        f.write(' ')
        f.write(str(mean_loss))
        f.write(' ')
        f.write(str(std_loss))


def visualize_tensor_in_out(input, output, save_path):
    with torch.no_grad():
        input = input.squeeze().cpu()
        output = output.squeeze().cpu()
    plt.plot(input, 'b')
    plt.plot(output, 'r')
    plt.fill_between(np.arange(256), output, input, color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.savefig(save_path)
    plt.cla()


def eval():
    if model_id == 0:
        model = autoencoder().to(device)
        save_dir = 'results/autoencoder/'
    else:
        model = convautoencoder().to(device)
        save_dir = 'results/convautoencoder/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + 'thresh.txt', 'r') as f:
        line = f.readline().split(' ')
        min_loss = float(line[0])
        max_loss = float(line[1])
        mean_loss = float(line[2])
        std_loss = float(line[3])
    print('min_loss:{}, max_loss:{}, mean_loss:{}, mean_loss:{}'.format(min_loss, max_loss, mean_loss, std_loss))
    threshold = mean_loss + std_loss

    test_dataset1 = AutoencoderDataset('./data/normal_test.csv')
    test_dataset2 = AutoencoderDataset('./data/abnormal_test.csv')
    print('Finish loading dataset, begin testing.')

    criterion = torch.nn.L1Loss(reduction='mean')

    model_path = save_dir + 'best.pth'
    model.load_state_dict(torch.load(model_path))

    # normal: P, abnormal: N

    all_loss = []
    TP, FP = 0, 0
    for i, data in enumerate(test_dataset1):
        with torch.no_grad():
            data = data.unsqueeze(0).to(device)
            pred = model(data)
            loss = criterion(pred, data).cpu().numpy()
            all_loss.append(loss)

            if i % 20 == 0:
                visualize_tensor_in_out(data, pred, save_dir + 'normal_test_' + str(i) + '.jpg')

            # if loss < min_loss or loss > max_loss:
            if loss > threshold:
                FP = FP + 1
            else:
                TP = TP + 1

    avg_loss = np.mean(all_loss)
    print('avg_loss of normal_test', avg_loss)

    all_loss = []
    TN, FN = 0, 0
    for i, data in enumerate(test_dataset2):
        with torch.no_grad():
            data = data.unsqueeze(0).to(device)
            pred = model(data)
            loss = criterion(pred, data).cpu().numpy()
            all_loss.append(loss)

            if i % 20 == 0:
                visualize_tensor_in_out(data, pred, save_dir + 'abnormal_test_' + str(i) + '.jpg')

            # if loss < min_loss or loss > max_loss:
            if loss > threshold:
                TN = TN + 1
            else:
                FN = FN + 1

    avg_loss = np.mean(all_loss)
    print('avg_loss of normal_test', avg_loss)

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f = 2 * p * r / (p + r)
    a = (TP + TN) / (TP + FP + TN + FN)
    print('precison:{}, recall:{}, f1:{}, accuracy:{}'.format(p, r, f, a))


if __name__ == '__main__':

    retrain = 0

    if retrain:
        train()
        cal_threshold()
        eval()
    else:
        eval()
