import os
import pandas as pd
import csv
import time

#数据文件夹，每个用户的csv
filePath ='test_train_folder/train_uid_folder9'#不同文件夹folder+i,i=1,2,...,
#存入地址
filepath1 = 'final_data/final_test_data.csv'
#读取文件
app_id_type = pd.read_csv('app_id_type.csv',encoding='utf-8',usecols=[1,2])
app_type_id = pd.read_csv('app_type_id.csv',encoding='utf-8',usecols=[1,2])

# 获取应用类型
def get_type(x):
    if len(app_id_type[app_id_type['app_id'].isin([x])]['type'].values)>0:
        return app_id_type[app_id_type['app_id'].isin([x])]['type'].values[0]
    else:
        str1 = '其它'
        return str1

# 写入csv文件
def storFile(fs_list,fileName):
    with open(fileName,'a',newline ='') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(fs_list)

#遍历用户csv 提取特征属性值
def readAllFiles(filePath):
    fileList = os.listdir(filePath)
    j=0
    for file in fileList:
        start=time.time()
        print(os.path.join(filePath,file))
        user_app_usage = pd.read_csv(os.path.join(filePath,file),usecols=[1,2,3,4])
        user_app_usage['type'] =[get_type(x) for x in user_app_usage['0_id']]
        fs_list=[]
        fs_list.append(file.replace(".csv",""))
        for i in app_type_id['type']:
            length = user_app_usage[user_app_usage['type'].str.contains(i)]
            if len(length)==0:
                fs_list.append(0)
                fs_list.append(0)
                fs_list.append(0)
            else:
                fs_list.append(length['1_use_duration'].sum())
                fs_list.append(length['2_open_times'].sum())
                fs_list.append(length['3_use_days'].max())
        qita = user_app_usage[user_app_usage['type'].str.contains('其它')]
        if len(qita)==0:
            fs_list.append(0)
            fs_list.append(0)
            fs_list.append(0)
        else:
            fs_list.append(qita['1_use_duration'].sum())
            fs_list.append(qita['2_open_times'].sum())
            fs_list.append(qita['3_use_days'].max())
        storFile(fs_list,filepath1)
        end=time.time()
        print('第{0}个文件夹,耗时:{1}'.format(j,end-start))
        j+=1

readAllFiles(filePath)