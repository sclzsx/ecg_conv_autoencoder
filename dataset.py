import os
import wfdb
from IPython.display import display
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def extraction(data_dir, save_dir, radius, extract_channel, extract_label, max_idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filenames = [i[:-4] for i in os.listdir(data_dir) if '.hea' in i]
    savedata = []
    savelabels = []
    for name in filenames:
        record = wfdb.rdheader(data_dir + '/' + name)
        display(record.__dict__)

        alldata = wfdb.rdrecord(data_dir + '/' + name).p_signal
        length, channel = alldata.shape
        print('length', length)
        if max_idx > length:
            max_idx = length
        elif max_idx < radius * 2 + 1:
            max_idx = length
        anno = wfdb.rdann(data_dir + '/' + name, 'atr')

        peaks = anno.sample
        for peak in peaks:
            start = peak - radius
            end = peak + radius
            if start < 0 or end > max_idx:
                continue
            for ch in range(channel):
                if extract_channel is not None:
                    if ch != extract_channel:
                        continue
                tmpdata = wfdb.rdrecord(data_dir + '/' + name, sampfrom=start, sampto=end, channels=[ch]).p_signal
                tmpdata = np.array(tmpdata).squeeze()
                tmplabel = wfdb.rdann(data_dir + '/' + name, 'atr', sampfrom=start, sampto=end).symbol
                # print(tmpdata.shape, len(tmplabel))
                if len(tmplabel) == 1:
                    tmplabel = tmplabel[0]
                    if extract_label is not None:
                        if tmplabel == extract_label:
                            # print(tmpdata.shape, tmplabel)
                            savedata.append(tmpdata)
                            savelabels.append(0)
                    else:
                        savedata.append(tmpdata)
                        savelabels.append(tmplabel)
        print(name, len(savedata), len(savelabels))
        # break
    savedata = np.array(savedata)
    savelabels = np.array(savelabels)
    print('all', savedata.shape)
    pd.DataFrame(savedata).to_csv(save_dir + '/data.csv', index=False, header=False)
    pd.DataFrame(savelabels).to_csv(save_dir + '/labels.csv', index=False, header=False)


def extraction2(data_dir, save_dir, radius, extract_channel, extract_label, max_idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filenames = [i[:-4] for i in os.listdir(data_dir) if '.hea' in i and 'hea-' not in i]
    # print(filenames)
    savedata = []
    savelabels = []
    for name in filenames:

        alldata = wfdb.rdrecord(data_dir + '/' + name).p_signal
        length, channel = alldata.shape
        print('length', length)
        if max_idx > length:
            max_idx = length
        elif max_idx < radius * 2 + 1:
            max_idx = length
        anno = wfdb.rdann(data_dir + '/' + name, 'ecg')
        # labels = anno.symbol
        peaks = anno.sample
        for peak in peaks:
            start = peak - radius
            end = peak + radius
            if start < 0 or end > max_idx:
                continue
            for ch in range(channel):
                if extract_channel is not None:
                    if ch != extract_channel:
                        continue
                tmpdata = wfdb.rdrecord(data_dir + '/' + name, sampfrom=start, sampto=end, channels=[ch]).p_signal
                tmpdata = np.array(tmpdata).squeeze()
                tmplabel = wfdb.rdann(data_dir + '/' + name, 'ecg', sampfrom=start, sampto=end).symbol
                # print(tmpdata.shape, len(tmplabel))
                if len(tmplabel) == 1:
                    tmplabel = tmplabel[-1]
                    # print(tmplabel)
                    if extract_label is not None:
                        if tmplabel == extract_label:
                            # print(tmpdata.shape, tmplabel)
                            savedata.append(tmpdata)
                            savelabels.append(1)
                    else:
                        savedata.append(tmpdata)
                        savelabels.append(tmplabel)
        print(name, len(savedata), len(savelabels))
        # break
    savedata = np.array(savedata)
    savelabels = np.array(savelabels)
    print('all', savedata.shape)
    pd.DataFrame(savedata).to_csv(save_dir + '/data.csv', index=False, header=False)
    pd.DataFrame(savelabels).to_csv(save_dir + '/labels.csv', index=False, header=False)


def normal_and_spit(normal_data_path, abnormal_data_path):
    normal_data = np.array(pd.read_csv(normal_data_path))
    abnormal_data = np.array(pd.read_csv(abnormal_data_path))

    print('before suffle', normal_data.shape)
    np.random.shuffle(normal_data)
    print('after shuffle', normal_data.shape)

    normal_num = len(normal_data)
    abnormal_num = len(abnormal_data)

    all_data = np.r_[normal_data, abnormal_data]
    min_val = np.min(all_data)
    max_val = np.max(all_data)

    normal_data = (normal_data - min_val) / (max_val - min_val)
    abnormal_data = (abnormal_data - min_val) / (max_val - min_val)

    normal_train = normal_data[:normal_num - abnormal_num, :]
    normal_test = normal_data[normal_num - abnormal_num:, :]

    pd.DataFrame(normal_train).to_csv('./data/normal_train.csv', index=False, header=False)
    pd.DataFrame(normal_test).to_csv('./data/normal_test.csv', index=False, header=False)
    pd.DataFrame(abnormal_data).to_csv('./data/abnormal_test.csv', index=False, header=False)


class AutoencoderDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.array(pd.read_csv(data_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = torch.tensor(self.data[i], dtype=torch.float32).unsqueeze(0)  # require shape [1 256]
        return data


if __name__ == '__main__':

    # # extract normal data
    # extract_channel = 0
    # radius = 128
    # max_idx = 650000
    # extract_label = 'N'
    # data_dir = './data/mit-bih-arrhythmia-database-1.0.0'
    # save_dir = './data/normal'
    # extraction(data_dir, save_dir, radius, extract_channel, extract_label, max_idx)

    # # extract abnormal data
    # extract_channel = 0
    # radius = 128
    # max_idx = 650000
    # extract_label = 'V'
    # data_dir = './data/files'
    # save_dir = './data/abnormal'
    # extraction2(data_dir, save_dir, radius, extract_channel, extract_label, max_idx)

    # normalize the data values to range (0,1)
    normal_and_spit('./data/normal/data.csv', './data/abnormal/data.csv')

    # check the dataset 
    train_dataset = AutoencoderDataset('./data/normal_test.csv')
    for i in train_dataset:
        print(i.shape)
        break
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    for i, batch_data in enumerate(train_dataloader):
        print(i, batch_data.shape)
