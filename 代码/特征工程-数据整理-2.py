'''
分类变量转化为数值变量
labelencoder
'''
#导入相应模块
'''
LabelEncoder:类别编码
'''
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

#读取数据
data_basic=pd.read_csv('user_basic_info.csv',header=None)

#重命名
data_basic_2=data_basic.rename(columns={0:'uid',1:'gender',2:'city',
                   3:'prodName',4:'ramCapacity',5:'ramLeftRation',
                   6:'romCapacity',7:'romLeftRation',8:'color',
                   9:'fontSize',10:'ct',11:'carrier',12:'os'})

#改变列表
change_list=['city','prodName','color','ct','carrier']

for col in change_list:
    try:
        data_basic_2[col]=label.fit_transform(data_basic_2[col].values)#转换数据
    except:
        data_basic_2[col]=label.fit_transform(data_basic_2[col].apply(str).values)

#保存数据
data_basic_2.to_csv('data_basic_info_change.csv',index=None)