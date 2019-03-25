import pandas as pd
import glob
paths = glob.glob('./training/*')

columns_name = ['FileName', 'TypeSepsis','StartTime','LenTime']
list_rows = []
type = [0,1]

for path in paths:
    file_name = path.split('/')[-1]
    df = pd.read_csv(path, delimiter='|')
    len_time = df.shape[0]

    flag = df[df['SepsisLabel']==1].drop_duplicates(subset=['SepsisLabel'])
    if flag.empty:
        row =  [file_name, type[0], -1, len_time]
        list_rows.append(row)
    else:
        start = flag.index[0]
        row = [file_name, type[1], start, len_time]
        list_rows.append(row)


save_file =  pd.DataFrame(list_rows, columns=columns_name).sort_values(by = ['FileName'], ascending= True)
path_save = './check.csv'
with open(path_save, 'w') as f:
    save_file.to_csv(f, encoding='utf-8', header=True, index = False)
print('Complete')


