import pandas as pd
import glob


def check_type(training_directory):
    paths = glob.glob('./' + training_directory + '/*')
    columns_name = ['FileName', 'TypeSepsis', 'Sex', 'Age', 'StartTime', 'LenTime']
    list_rows = []
    type = [0, 1]
    for path in paths:
        file_name = path.split('/')[-1]
        df = pd.read_csv(path, delimiter='|')
        len_time = df.shape[0]
        flag = df[df['SepsisLabel'] == 1].drop_duplicates(subset=['SepsisLabel'])
        age = df['Age'].iloc[0]
        gender = df['Gender'].iloc[0]
        if flag.empty:
            row = [file_name, type[0], gender, age - 1, len_time]
            list_rows.append(row)
        else:
            start = flag.index[0]
            row = [file_name, type[1], gender, age, start, len_time]
            list_rows.append(row)

    save_file = pd.DataFrame(list_rows, columns=columns_name).sort_values(by=['FileName'], ascending=True)
    path_save = './check.csv'
    with open(path_save, 'w') as f:
        save_file.to_csv(f, encoding='utf-8', header=True, index=False)
    print('Complete check type')


def convert_to_csv(training_directory, training_directory_csv):
    paths = glob.glob('./' + training_directory + '/*')
    for path in paths:
        file_name = path.split('/')[-1]
        df = pd.read_csv(path, delimiter='|')
        path_save = './' + training_directory_csv + '/' + file_name
        with open(path_save, 'w') as f:
            df.to_csv(f, encoding='utf-8', header=True, index=False)


def main():
    name_dic = 'training'
    name_dic_csv = 'training_csv'
    check_type(name_dic)
    convert_to_csv(name_dic, name_dic_csv)

main()