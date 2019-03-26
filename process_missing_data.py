import pandas as pd
import  os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

path_file = './summary.csv'

train =  pd.read_csv(path_file)

train.info()
# # print(train)
#
# patients_sepsis = train[train['SepsisLabel']==1]
# patients_sepsis.info()
# # print(train)
# colormap = plt.cm.RdBu
# plt.figure(figsize=(32,32))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# img = sns.heatmap(patients_sepsis.corr(),linewidths=0.1,vmax=1.0,
#             square=True, cmap=colormap, linecolor='white', annot=True)
#
# figure = img.get_figure()
# figure.savefig('svm_conf.png', dpi=100)
# Note that the categorical features have been neglected in the
# correlation matrix.
