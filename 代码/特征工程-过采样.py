
#导入相关模块
'''
SMOTE:过采样模块
'''

import pandas as pd
from imblearn.over_sampling import SMOTE

#读取数据
data=pd.read_csv('new_train_data_2.csv',index_col=0)
#删除缺失值
data_1=data.dropna()
#构建过采样比例
smote=SMOTE(ratio={1:260000,6:190000},random_state=42,n_jobs=4)#label：1扩充数据260000，label：6扩充数据190000

#过采样
x,y=smote.fit_sample(data_1.iloc[:,1:].drop('target_age',axis=1),data_1.target_age)

#提取标签为1的x扩充数据
data_value_1=[list(x[i])+[1] for i in range(len(y)) if y[i]==1]

#提取标签为6的x扩充数据
data_value_6=[list(x[i])+[6] for i in range(len(y)) if y[i]==6]

#合并数据
data_new=data_value_1+data_value_6
cols=data_1.iloc[:,1:].drop('target_age',axis=1).columns#获取列名
cols=list(cols)+['target_age']#扩充标签列名
data_new_df=pd.DataFrame(data_new,columns=cols)
data_2=data.drop('target_age',axis=1)
data_2['target_age']=data.target_age
data_2=data_2.drop('uid',axis=1)
new_data=pd.concat([data_2,data_new_df],axis=0,ignore_index=True)#拼接原始数据和扩充数据
df=new_data.sample(frac=1)
df.to_csv('train_oversample.csv')#保存过采样数据