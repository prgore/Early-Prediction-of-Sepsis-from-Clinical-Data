import pandas as pd
import os

path_folder = './training'
list_files = pd.read_csv('./check.csv')
list_files =  list_files[list_files['TypeSepsis']==1]
num_file = list_files.shape[0]
for i in range(num_file):
    file = os.path.join(path_folder, list_files.iloc[i]['FileName'])
    df = pd.read_csv(file, delimiter='|')
    # print(df)
    if i ==0:
        frames = df
    else:
        frames = [frames, df]
        frames = pd.concat(frames)


path_save = './summary_1.csv'
with open(path_save, 'w') as f:
    frames.to_csv(f, encoding='utf-8', header=True, index = False)



