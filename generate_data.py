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
def divide_data(sepsis=0, gender=0, age=0, interpolation=False):
    #Name for save model
    name_folder = str(sepsis)+str(gender)+str(age)

    #Divide data to group
    df = pd.read_csv(path_check_file)
    df = df[df['TypeSepsis'] == sepsis]
    df = df[df['Sex'] == gender]
    if age == 0:
        df = df[df['Age'] <= 50]
    else:
        df = df[df['Age'] > 50]

    #List file in the group
    file_names = df['FileName']

    #Concatenate all file to a frame
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


    return frames, name_folder


def process_string_info_frame(frames):
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

    return num, temp


def neural_network(x, y, name_feature, folder_model_sub):
    #Create dataset
    x_train,x_val,y_train,y_val = train_test_split(x, y, test_size= 0.3)

    #Set parameters
    units = 512
    epochs = 200
    batch_size = 4096

    #Create model
    shape_input = str(x_train.shape[1])
    input_model = Input(shape=(x_train.shape[1],))
    # x = Flatten()(input_model)
    x = Dense(units)(input_model)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    output_model = Dense(1, activation='linear')(x)
    model = Model(inputs=input_model, outputs=output_model)
    model.summary()

    #Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    # Reducer Learning rate
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

    # Check point save all model
    folder =  folder_model + "/" + folder_model_sub
    cf.create_folder(folder)
    name_file = shape_input + "_" + name_feature + ".hdf5"
    filepath = folder + "/" + name_file
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    '''
    #Tensor board
    tensor_board = TensorBoard(log_dir=path_folder_gen_model, histogram_freq=0, write_graph=True, write_images=True)
    '''
    callbacks_list = [lr_reducer, checkpoint]

    # Fit model
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
              callbacks=callbacks_list, verbose=2, shuffle=True)

    return filepath


def process_missing_data(frames, folder_save,neural=True):
    frames = frames
    while (True):
        # Processing string
        num, temp = process_string_info_frame(frames)

        # Use linear
        items_full = temp[temp['None-Null'] == num]['Name'].values.tolist()
        items_missing = temp[temp['None-Null'] < num]['Name']
        num_items_missing = items_missing.shape[0]

        if num_items_missing == 0:
            break

        items_name = items_missing.iloc[0]
        items_full.append(items_name)
        data = frames[items_full]
        print(items_name)
        # Step-1: Split the dataset that contains the missing values and no missing values are test and train respectively.
        x_train = data[data[items_name].notnull()].drop(columns=items_name)
        y_train = data[data[items_name].notnull()][items_name]
        x_test = data[data[items_name].isnull()].drop(columns=items_name)

        # Step-2: Train and prediction the machine learning algorithm
        if neural:
            # Predict with best model
            filepath = neural_network(x_train, y_train, items_name, folder_save)
            best_model = load_model(filepath)
            predicted = best_model.predict(x_test, batch_size=512)
        else:
            linreg = LinearRegression()
            linreg.fit(x_train, y_train)
            predicted = linreg.predict(x_test)

        # Step-4: Let’s obtain the complete dataset by combining with the target attribute.
        frames[items_name][frames[items_name].isnull()] = predicted

    return frames


def split_to_object(frames, folder_data='data'):
    name_files = frames.drop_duplicates(subset=['FileName'], keep='first')['FileName']
    num_files = name_files.shape[0]
    for i in range(num_files):
        name_file = name_files.iloc[i]
        temp = frames[frames['FileName'] == name_file]
        folder_save = './' + folder_data
        cf.create_folder(folder_save)
        path_save = folder_save+ '/' + name_file
        with open(path_save, 'w') as f:
            print(path_save)
            temp.to_csv(f, encoding='utf-8', header=True, index=False)


def main():

    df, name_folder = divide_data(1,1,1)
    frames = process_missing_data(df, name_folder)
    split_to_object(frames)



main()
# def process_missing_for_test():
