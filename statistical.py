import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

file_sum = './summary.csv'
df = pd.read_csv(file_sum)
name_columns = list(df)
len = len(name_columns)
label = name_columns [-1]
for i in range(len-1):
    df_temp = df[[name_columns[i], label]]
    df_temp = df_temp.dropna()

    df_normal = df_temp [df_temp[label]==0].mean()
    print(df_normal)
    df_sepsis = df_temp [df_temp[label]==1].mean()
    print(df_sepsis)
    print('++++++++++++++++++++++++++++++++++++++++')
    # img1 = sns.jointplot(x =name_columns[i], y = label, data = df_normal)
    # img2 = sns.jointplot(x=name_columns[i], y=label, data=df_sepsis)
    #
    # # img = df_temp.plot.scatter(x =name_columns[i], y = label, c = 'Red')
    # img1.savefig('./visualize/'+str(name_columns[i])+'_normal.png')
    # img2.savefig('./visualize/' + str(name_columns[i]) + '_sepsis.png')
    #
    # print(i)