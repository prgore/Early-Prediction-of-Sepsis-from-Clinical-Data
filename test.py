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
        df = df[df['Age'] >= 50 ]
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

def process_missing_data(frames):
    buf  = io.StringIO()
    frames.info(buf = buf)
    s = buf.getvalue()
    list = s.split('\n')[3:-5]
    label = ['Name', 'None-Null', 'Temp' ,'Type']
    temp =[]
    for s in list:
        s = re.sub(' +',' ', s)
        array = s.split(' ')
        temp.append(array)

    df = pd.DataFrame(data=temp, columns=label)
    df = df.drop('Temp', axis=1)
    # df = df.sort_values(by =['Non'])
    df['None-Null'] = df['None-Null'].astype(int)
    df = df.sort_values(by = ['None-Null'], ascending= False)
    top_items =  df.head(20)
    print(top_items)


    linreg = LinearRegression()
    # print(top_items)
    data = frames[top_items['Name']]
    # print(data)
process_missing_data(create_data_frame(get_file_name()))