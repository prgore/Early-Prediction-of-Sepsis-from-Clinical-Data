import pandas as pd
import numpy as np
import glob
path_folder = './data/*'

def gen_data(sample):
    sample = sample.loc[:, sample.columns != 'FileName']
    label = sample.loc[:, sample.columns=='SepsisLabel'].values
    num_times =  sample.shape[0]


    padding = pd.DataFrame(np.tile(sample.iloc[-1], (47, 1)), columns=list(sample.columns.values))
    temp = sample.append(padding)
    sample = temp.loc[:, temp.columns!='SepsisLabel'].values

    list_sample = list()
    stride_window = 48
    for i in range(num_times):
        x = sample [i:i+stride_window]
        list_sample.append(x)

    sample = np.array(list_sample)
    return sample, label

def read_data():
    paths = glob.glob(path_folder)
    samples = list()
    labels = list()
    for path in paths:
        data_raw  = pd.read_csv(path)
        sample, label = gen_data(data_raw)
        samples.append(sample)
        labels.append(label)
    samples = np.vstack(samples)
    print(samples.shape)
    labels = np.vstack(labels)
    print(labels.shape)
    return samples, labels


read_data()
