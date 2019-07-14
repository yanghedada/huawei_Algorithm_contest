
#导入相关模块
'''
PCA:主成分分析模块
FontProperties:画图字体类
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#设置字体以便显示中文
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

#读取数据
data=pd.read_csv('app_property_train.csv',header=None,error_bad_lines=False)
data_martix=data.dropna()#删除缺失值
data_martix1=data_martix.iloc[:,1:]#除第0列id外均降维

explained_variance_ratio_=[]#解释百分比
explained_variance_=[]#解释方差
n_components=list(range(1,31))#观察降维特证数列表

#不同保留特征数PCA降维表现
for i in range(1,31):
    pca=PCA(n_components=i,random_state=3)
    pca.fit_transform(data_martix1)
    explained_variance_ratio_.append(pca.explained_variance_ratio_.sum())
    explained_variance_.append(pca.explained_variance_.sum())

#降维效果可视化
fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(20,15))
ax1,ax2=axes.ravel()
plt.tight_layout(pad=6)
ax1.plot(n_components,explained_variance_ratio_,'g-')
ax1.hlines(y=0.95,xmin=1,xmax=7,colors='r',linestyles='dashed',alpha=0.7,label='95%')
ax1.hlines(y=0.9,xmin=1,xmax=7,colors='b',linestyles='dotted',alpha=0.7,label='90%')
ax1.legend(loc=4)
ax1.set_xlabel('维数',FontProperties=font)
ax1.set_ylabel('累计方差贡献率',FontProperties=font)
ax1.set_title('累计方差贡献率变化图',FontProperties=font)
ax2.plot(n_components,explained_variance_,'r-')
ax2.set_xlabel('维数',FontProperties=font)
ax2.set_ylabel('累计方差',FontProperties=font)
ax2.set_title('累计方差变化图',FontProperties=font)

pca_30=PCA(n_components=30,random_state=3)#经观察保留特征设置为30
data_martix2=pca_30.fit_transform(data_martix1)#PCA降维转换数据

#重构数据集成Dataframe形式
data_martix3=pd.DataFrame(data_martix2,columns=['app_property_'+str(i) for i in range(1,31)])
data_martix3['uid']=data_martix.uid.values
data_martix3.to_csv('app_property_train_pca.csv')#保存降维后数据
