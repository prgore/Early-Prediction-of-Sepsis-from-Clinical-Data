import pandas as pd
import numpy as np
import glob

path_folder = './test_1_1_1/*'



def read_data():
    paths = glob.glob(path_folder)
    # print(paths)
    samples = []
    labels = []

    for path in paths:
        data = pd.read_csv(path)
        sample = data.loc[:, data.columns != 'FileName'].values
        len_timestep = sample.shape[0]
        if (len_timestep < 48):
            padding = np.tile(sample[-1], ((48 - len_timestep), 1))
            sample = np.append(sample, padding, axis=0)
            x =  sample[:,:-1]
            y = sample[:,-1]
            samples.append(x)
            labels.append(y)

        elif (len_timestep == 48):
            x = sample[:, :-1]
            y = sample[:, -1]
            samples.append(x)
            labels.append(y)
        else:
            sample = sample[-48:]
            x = sample[:, :-1]
            y = sample[:, -1]
            samples.append(x)
            labels.append(y)

    samples = np.array(samples)
    print(samples.shape)
    labels =  np.array(labels)
    print(labels.shape)
    return samples, labels

read_data()
