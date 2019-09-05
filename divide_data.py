import pandas as pd
from shutil import copyfile, move

data_info = pd.read_csv('./check_setB.csv')
path_data = './data/'
# print(data_info)
sepsis_files =  data_info[data_info['TypeSepsis']==1]
non_sepsis_files =  data_info[data_info['TypeSepsis']==0]

len_sepsis_file =  sepsis_files.shape[0]
num_train_sepsis = int(len_sepsis_file* 0.7)

len_non_sepsis_file = non_sepsis_files.shape[0]
num_train_non_sepsis =  int(len_non_sepsis_file*0.7)


path_train = './train/'
path_test = './test/'
#Divide sepsis file
for i in range(len_sepsis_file):
    row = sepsis_files.iloc[i]
    file_name = row['FileName']
    src = path_data + file_name
    if i <= num_train_sepsis:
        dest = path_train + file_name
        copyfile(src, dest)
    else:
        dest =  path_test + file_name
        copyfile(src, dest)

#Divide non-sepsis file
for i in range(len_non_sepsis_file):
    row = non_sepsis_files.iloc[i]
    file_name = row['FileName']
    src = path_data + file_name
    if i <= num_train_non_sepsis:
        dest = path_train + file_name
        copyfile(src, dest)
    else:
        dest =  path_test + file_name
        copyfile(src, dest)



# def divide_data():
#     #Set path
#     normal_path = './data/normal/'
#     aki_path = './data/aki/'
#     timeseries_path = '/media/HDD/quanglv/MIMIC-III/Direction3/chart_timeseries'
#     hadm, hadm_aki = filter_hadm()
#
#     #Write to csv
#     hadm['LABEL'] = 0
#     hadm.loc[hadm.HADM_ID.isin(hadm_aki.HADM_ID), 'LABEL'] =1
#     label_0 = hadm[hadm.LABEL == 0] #36452
#     label_1 = hadm[hadm.LABEL == 1] #9356
#
#     #Move file to folder suitable
#     #Normal folder
#     for row in label_0.iterrows():
#         name_file =  str(row[1].SUBJECT_ID)+'_'+ str(row[1].HADM_ID)+'.csv'
#         src_file = os.path.join(timeseries_path, name_file)
#         dest_file = os.path.join(normal_path, name_file)
#         if os.path.exists(src_file):
#             # copyfile(src, dst)
#             copyfile(src_file, dest_file)
#
#     #AKI folder
#     for row in label_1.iterrows():
#         name_file =  str(row[1].SUBJECT_ID)+'_'+ str(row[1].HADM_ID)+'.csv'
#         src_file = os.path.join(timeseries_path, name_file)
#         dest_file = os.path.join(aki_path, name_file)
#         if os.path.exists(src_file):
#             # copyfile(src, dst)
#             copyfile(src_file, dest_file)