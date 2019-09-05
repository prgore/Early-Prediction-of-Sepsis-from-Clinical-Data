import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import os
import time
# import read_data as rd
import test_2 as rd
from random import shuffle

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Activation, Dropout, Input, Masking, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE

# Setup GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Config minimize GPU with model
config_gpu = False
if config_gpu:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

# Training parameter
epochs = 50
batch_size = 1024
timesteps = 48
data_dim = 40
units = 256
dropout = 0.4
output = 1
layers = 2

results_list = []


# Read data
# -----------Train set-----------
samples, labels = rd.read_data()
num_samples = samples.shape[0]
index = np.arange(num_samples)
np.random.shuffle(index)
samples = samples[index]
labels = labels[index]
print(len(labels[labels ==1]))
print(len(labels[labels==0]))


train_X = samples[:int(num_samples*0.7)]
train_y = labels[:int(num_samples*0.7)]

# -----------Val set-----------
temp_X = samples[int(num_samples*0.7):]
temp_y = labels[int(num_samples*0.7):]

val_X = temp_X[:int(temp_X.shape[0]*0.5)]
val_y = temp_y[:int(temp_y.shape[0]*0.5)]

#------------Test set-----------
test_X = temp_X[int(temp_X.shape[0]*0.5):]
test_y = temp_y[int(temp_y.shape[0]*0.5):]

# Build model
input_model = Input(shape=(timesteps, data_dim))
x = Masking()(input_model)
x = LSTM(units, return_sequences=True)(x)
x = Dropout(dropout)(x)
x = LSTM(units, return_sequences=True)(x)
x = Dropout(dropout)(x)
x = LSTM(units)(x)

result = Dense(output, activation='sigmoid')(x)

model = Model(inputs=input_model, outputs=result)
# Summary
model.summary()

# Compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

# Reducer Learning rate
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
#
# # early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
#
# # Check point save all model
# filepath = directory_model + "/weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
#
# # Tensor board
# tensor_board = TensorBoard(log_dir=directory_logs, histogram_freq=0, write_graph=True, write_images=True)
#
# # Results
# results = Results(val_X, val_y, batch_size, results_list)
#
# List callbacks
callbacks_list = [lr_reducer]
#
# # fit network
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=2,
                    shuffle=True, callbacks=callbacks_list)
#
# write_log(file_name, results_list)
#
# make a prediction
yhat = model.predict(test_X, batch_size=batch_size)
print(yhat)
# yhat[yhat>0.5] = 1
# yhat[yhat<=0.5] = 0
#
print(yhat)
# print('++++++++++++++++++++++++++++=+++++++++++++++++')
# print(test_y)

