import pandas  as pd
import glob
paths = glob.glob('./training/*')
for path in paths:
    file_name = path.split('/')[-1]
    df = pd.read_csv(path, delimiter='|')
    path_save = './csv_file/'+ file_name
    with open(path_save, 'w') as f:
        df.to_csv(f, encoding='utf-8', header=True, index = False)