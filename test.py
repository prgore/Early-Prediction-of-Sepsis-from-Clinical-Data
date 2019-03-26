import pandas as pd
import os
import io
import re
from sklearn.linear_model import LinearRegression

path_check_file = './check.csv'
path_folder = './training'


#sepsis with 0 is normal, 1 is sepsis
#gender= [0,1] with 0 is male, 1 is female
#thredhold of default age is 50 with True is age > 50 and False is age <=50
def get_file_name(sepsis = 0, gender = 0, age = 0):
    df = pd.read_csv(path_check_file)
    df = df[df['TypeSepsis']==sepsis]
    df = df[df['Sex'] == gender]
    if age == 0:
        df = df[df['Age'] <= 50 ]
    else:
        df = df[df['Age'] > 50 ]
    return df

def create_data_frame(df_file_name, interpolation = False):
    file_names = df_file_name['FileName']
    len  = file_names.shape[0]
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
    frames.info(buf = buf)
    s = buf.getvalue()
    num = int(s.split('\n')[1].split(' ')[1])
    list = s.split('\n')[3:-5]
    label = ['Name', 'None-Null', 'Temp' ,'Type']
    temp =[]
    for s in list:
        s = re.sub(' +',' ', s)
        array = s.split(' ')
        temp.append(array)

    temp = pd.DataFrame(data=temp, columns=label)
    temp = temp.drop('Temp', axis=1)
    temp['None-Null'] = temp['None-Null'].astype(int)
    temp = temp.sort_values(by = ['None-Null'], ascending= False)

    return num, temp


def process_missing_data(frames):
    frames = frames
    while(True):
        #Processing string
        num, temp = process_string_info_frame(frames)

        #Use linear
        items_full = temp[temp['None-Null']==num]['Name'].values.tolist()
        items_missing = temp[temp['None-Null']< num]['Name']
        num_items_missing = items_missing.shape[0]
        if num_items_missing == 0:
            break

        items_name = items_missing.iloc[0]
        items_full.append(items_name)

        linreg = LinearRegression()
        data = frames[items_full]
            
        
        #Step-1: Split the dataset that contains the missing values and no 
        # missing values are test and train respectively.
        x_train = data[data[items_name].notnull()].drop(columns= items_name)
        y_train = data[data[items_name].notnull()][items_name]
        x_test = data[data[items_name].isnull()].drop(columns=items_name)
        y_test = data[data[items_name].isnull()][items_name]

        #Step-2: Train the machine learning algorithm
        linreg.fit(x_train, y_train)

        #Step-3: Predict the missing values in the attribute of the test data.
        predicted = linreg.predict(x_test)

        #Step-4: Let’s obtain the complete dataset by combining with the target attribute.
        frames[items_name][frames[items_name].isnull()] = predicted
        # frames.info()
        print(frames)
    # path_save = './xxxxxxxxxxxx.csv'
    # with open(path_save, 'w') as f:
    #     frames.to_csv(f, encoding='utf-8', header=True, index = False)
    # return frames

process_missing_data(create_data_frame(get_file_name()))
