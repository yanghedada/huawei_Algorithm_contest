
#导入相关模块
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#读取数据
train=pd.read_csv('train_data.csv')

#类别是否均匀分布,不均匀后面要分层抽样
sns.set(style="darkgrid")
sns.countplot(train.target_age)

#计算相关系数

#准备数据
data=train.iloc[:,1:]
#获取类别
cols=data.columns
#计算相关系数
data_corr=data.corr().abs()

#相关系数热力图
plt.figure(figsize=(30,30))
sns.heatmap(data_corr,cbar=True,annot=True,linecolor='white',linewidths =0.1,cmap='summer')

#列出相关系数0.5以上的特征
threshold = 0.5
corr_list = []

for i in range(0,21): #特征，n的数字由想计算的相关系数决定
    for j in range(i+1,21): #
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j])
#强相关特征排序          
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
    print(stats.pearsonr(martix[i],martix[j]))#相关系数大于0.5的特征间是否显著

#性别分布
sns.set(style="darkgrid")
sns.countplot(train.target_age,order=train.gender)

#city分布
sns.set(style="darkgrid")
sns.pairplot(train,x_vars='city',y_vars='target_age')

#标签关于性别是否平衡
ax=plt.gca()
ax.bar([1,2,3,4,5,6],gender_by_age.values[1::2],label='0')
ax.bar([1,2,3,4,5,6],-gender_by_age.values[::2],label='1')
plt.legend()

#删除空列
drop_list=[]
for i in data_app_col:
    if len(train[i].unique())==1:
        drop_list.append(i)
        print(str(i)+'-unique:{}'.format(train[i].unique()))#唯一值为0

#统计缺失值数目
for col in train_data.iloc[:,1:].columns:
        print(col+'--缺失值数目:{0}\n--缺失值数目占比:{1}%'
	    .format(len(train[col])-train[col].count(),
	    (len(train[col])-train[col].count())/len(train[col])*100))