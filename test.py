import pandas as pd 
import glob

paths = glob.glob('./training/*')

for path in paths:
    print(path)
    data = pd.read_csv(path,delimiter ='|')
    print(data)
