import pandas as pd
import numpy as np
import glob
path_folder = './train/*'

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
    num_batch = int(len(paths)/10)
    for i in range(100):
        if i == 99:
            sub_paths = path_folder[i*num_batch:]
        sub_paths = path_folder[i*num_batch :(i+1)*num_batch]
        samples = list()
        labels = list()
        for path in sub_paths:
            data_raw  = pd.read_csv(path)
            sample, label = gen_data(data_raw)
            samples.append(sample)
            labels.append(label)
        samples = np.vstack(samples)
        print(samples.shape)
        labels = np.vstack(labels)
        print(labels.shape)
        num_samples = samples.shape[0]
        index = np.arange(num_samples)
        np.random.shuffle(index)
        samples = samples[index]
        labels = labels[index]
        yield (samples, labels)
