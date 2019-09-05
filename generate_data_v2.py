import pandas as pd
from os import listdir
import os
import io
import re
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Input, BatchNormalization, Flatten
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
import keras
import create_folder as cf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Config minimize GPU
config_gpu = True
if config_gpu:
    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

def process_string_info_frame(frames):
    buf = io.StringIO()
    frames.info(buf=buf)
    s = buf.getvalue()
    num = int(s.split('\n')[1].split(' ')[1])
    list = s.split('\n')[3:-5]
    label = ['Name', 'Not Null', 'Count', 'Type']
    temp = []
    for s in list:
        s = re.sub(' +', ' ', s)
        array = s.split(' ')
        temp.append(array)

    temp = pd.DataFrame(data=temp, columns=label)
    temp = temp.drop('Count', axis=1)
    temp['Not Null'] = temp['Not Null'].astype(int)
    temp = temp.sort_values(by=['Not Null'], ascending=False)

    return temp


def generate_data(folder_data,interpolation=False):
    #Divide data to group
    list_paths =  listdir(folder_data)
    list_paths.sort()
    length =  len(list_paths)
    num_sub_paths =  int(length/100)
    for i in range(2):
        if i == num_sub_paths:
            sub_paths = list_paths[i*100 :]
        sub_paths =  list_paths[i*100: (i+1)*100]
        for path in sub_paths:
            df = pd.read_csv(os.path.join(folder_data,path), delimiter='|')
            if interpolation == True:
                df = df.interpolate(method='linear').ffill().bfill()
            df = df.ffill().bfill()
            if i == 0:
                frames = df
            else:
                frames = [frames, df]
                frames = pd.concat(frames)
        temp = process_string_info_frame(frames)
        if i ==0:
            temps = temp
        temps = [temps, temp]
        temps = pd.concat(temps)
    temps['Sum']= temps.groupby(['Name'])['Not Null'].agg('sum')
    # temps = temps.sort_values(by=['Sum'], ascending= True)
    print(temps)
    # list_column = count.values
    # print(list_column)

    # print(a)
    # print(type(a))
        # print(temp)

generate_data('./training_setA')