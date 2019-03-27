import pandas as pd
import os
import io
import re
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Input, BatchNormalization, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import keras

#Config minimize GPU
config_gpu = True
if config_gpu:
    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

#Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path_check_file = './check.csv'
path_folder = './training'


# sepsis with 0 is normal, 1 is sepsis
# gender= [0,1] with 0 is male, 1 is female
# thredhold of default age is 50 with True is age > 50 and False is age <=50
def get_file_name(sepsis=0, gender=0, age=0):
    df = pd.read_csv(path_check_file)
    df = df[df['TypeSepsis'] == sepsis]
    df = df[df['Sex'] == gender]
    if age == 0:
        df = df[df['Age'] <= 50]
    else:
        df = df[df['Age'] > 50]
    return df


def create_data_frame(df_file_name, interpolation=False):
    file_names = df_file_name['FileName']
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
    return frames


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


def neural_network(x_train_, y_train_, x_test):
    num_examples =  x_train_.shape[0]
    x_train = x_train_[:int(num_examples*0.7)]
    y_train = y_train_[:int(num_examples*0.7)]
    x_val = x_train_[int(num_examples*0.7):]
    y_val = y_train_[int(num_examples*0.7):]

    units = 128
    epochs = 100
    batch_size = 128

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
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    # Reducer Learning rate
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

    callbacks_list = [lr_reducer]
    # ------Fit network---------
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
              callbacks=callbacks_list, verbose=2, shuffle=True)
    y_test = model.predict(x_test, batch_size=batch_size)
    return y_test


def process_missing_data(frames, neural = True):
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

        # Step-1: Split the dataset that contains the missing values and no
        # missing values are test and train respectively.
        x_train = data[data[items_name].notnull()].drop(columns=items_name)
        y_train = data[data[items_name].notnull()][items_name]
        x_test = data[data[items_name].isnull()].drop(columns=items_name)
        # y_test = data[data[items_name].isnull()][items_name]
        # print(y_test)
        if neural:
            predicted = neural_network(x_train, y_train, x_test)
        else:
            linreg = LinearRegression()
            # Step-2: Train the machine learning algorithm
            linreg.fit(x_train, y_train)

            # Step-3: Predict the missing values in the attribute of the test data.
            predicted = linreg.predict(x_test)

        # Step-4: Let’s obtain the complete dataset by combining with the target attribute.
        frames[items_name][frames[items_name].isnull()] = predicted
        # print(frames)
        # frames.info()
    # print(frames)
    # path_save = './xxxxxxxxxxxx.csv'
    # with open(path_save, 'w') as f:
    #     frames.to_csv(f, encoding='utf-8', header=True, index = False)
    return frames


def split_to_object(frames):
    # print(frames)
    name_files = frames.drop_duplicates(subset=['FileName'], keep='first')['FileName']
    num_files = name_files.shape[0]
    for i in range(num_files):
        name_file = name_files.iloc[i]
        temp = frames[frames['FileName'] == name_file]
        path_save = './test/' + name_file
        with open(path_save, 'w') as f:
            temp.to_csv(f, encoding='utf-8', header=True, index=False)


split_to_object(process_missing_data(create_data_frame(get_file_name())))
