import pandas as pd
import os
import io
import re

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

# Set path
path_check_file = './check_setB.csv'

path_folder = './training_setB'
cf.create_folder(path_folder)

folder_model = './generate_model'
cf.create_folder(folder_model)

'''
Sepsis with 0 is normal, 1 is sepsis
Gender= [0,1] with 0 is male, 1 is female
Thredhold of default age is 50 with True is age > 50 and False is age <=50
'''
def process_missing_data(sepsis=0, gender=1, age=0, interpolation=False):

    # Divide data to group
    df = pd.read_csv(path_check_file)
    df = df[df['TypeSepsis'] == sepsis]
    df = df[df['Sex'] == gender]
    if age == 0:
        df = df[df['Age'] <= 50]
    else:
        df = df[df['Age'] > 50]

    # List file in the group
    file_names = df['FileName']

    # Concatenate all file to a frame
    len = file_names.shape[0]
    for i in range(len):
        file = os.path.join(path_folder, file_names.iloc[i])
        df = pd.read_csv(file, delimiter='|')
        if interpolation == True:
            df = df.interpolate(method='linear').ffill().bfill()
        df = df.ffill().bfill()
        df['FileName'] = file_names.iloc[i]
        if i == 0:
            frames = df
        else:
            frames = [frames, df]
            frames = pd.concat(frames)

    #Process string information of frame
    buf = io.StringIO()
    frames.info(buf=buf)
    s = buf.getvalue()
    num = int(s.split('\n')[1].split(' ')[1])
    list = s.split('\n')[3:-5]
    label = ['Name', 'None-Null', 'Temp', 'Type']
    temp = []
    for s in list:
        s = re.sub(' +', ' ', s)
        array = s.split(' ')
        temp.append(array)

    temp = pd.DataFrame(data=temp, columns=label)
    temp = temp.drop('Temp', axis=1)
    temp['None-Null'] = temp['None-Null'].astype(int)
    temp = temp.sort_values(by=['None-Null'], ascending=False)
    print(temp)

x_ =[0,1]
y_ = [0,1]
z_ = [0,1]

for x in x_:
    for y in y_:
        for z in z_:
            process_missing_data(x,y,z)