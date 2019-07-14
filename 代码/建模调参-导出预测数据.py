'''
同前面调节类似，改动其中参数部分即可，依次调节以下参数：
1.learning_rate：过拟合降低此值，欠拟合加大该值，0-1之间
2.subsample和colsample_bytree:即行采样和列采样,0-1之间
3.reg_lambda,reg_alpha:L2和L1正则化系数，防止过拟合
4基于以上参数继续微调n——setimators
'''
#import导入相关模块
'''
XGBClassifier:sklearn中的xgboost分类器
xgb:xgb库中自带xgb分类器
pandas:数据分析模块
numpy:矩阵计算模块
GridSearchCV:网格搜索类
StratifiedKFold:K折交叉验证类
log_loss:log_loss损失函数
pyplot:画图模块
sns:画图模块
accuracy_score：预测分数类
confusion_matrix：混淆矩阵类
time：时间模块
'''
import xgboost as xgb
from xgboost import XGBClassifier as XGB

import numpy as np 
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import time

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
%matplotlib inline

#读取数据
train_data=pd.read_csv('train_new.csv',index_col=0)#训练集
test_data=pd.read_csv('test_new.csv',index_col=0)#验证集，即要提交结果的那部分

#训练集处理
X=train_data.drop(['target_age','uid'],axis=1)#去除训练数据中的目标变量和id变量
Y=train_data.target_age#提取训练数据中的目标变量

def pre_result(n_estimators,max_depth,min_child_weight,test_data):#只列出这三个参数，其他参数函数内调整
    xgbmodel=XGB(max_depth=max_depth,
                 n_estimators=n_estimators,
                 silent=False,
                 n_jobs=8,#多线程跑模型
                 min_child_weight=min_child_weight,
                 random_state=3,
                 gpu_id=0,#如果有GPU，GPU序号
                 tree_method='gpu_hist',#采用GPU
                 objective='multi:softmax',
                 predictor='cpu_predictor')#防止GPU内存不足，采用CPU预测
    start=time.time()
    xgbmodel.fit(X,Y,eval_metric='mlogloss')#训练模型
    end=time.time()
    print('{0}-{1}-{2}-time:{3}'.format(n_estimators,max_depth,min_child_weight,end-start))#记录模型运行时间
    xtrain_pre=xgbmodel.predict(X)
    x_train_score=accuracy_score(Y,xtrain_pre)
    end=time.time()
    print(end-start,'--训练集')#记录训练集预测运行时间
    print('训练集分数是:{}%'.format(x_train_score*100))
    test_uid=test_data[['uid','gender']]
    test=test_data.drop(['uid'],axis=1)
    test_pre=xgbmodel.predict(test)
    test_uid['label']=test_pre#构建输出结果格式
    end=time.time()
    print(end-start,'--验证集')
    test_uid=test_uid.drop('gender',axis=1)
    test_done=pd.read_csv('test_done.csv')
    test_uid=pd.concat([test_uid,test_done],ignore_index=True)#合并缺少特征数据部分已经预测的结果
    test_uid.to_csv('{0}-{1}-{2}-final-submission.csv'.format(n_estimators,max_depth,min_child_weight),index=False)
    print('{0}-{1}-{2}-submission.csv存储完成!'.format(n_estimators,max_depth,min_child_weight))
    end=time.time()
    print(end-start)

if __name__=='__main__':
    proerty_list=[]#如果有多组n_estimators,max_depth,min_child_weight参数，否则直接调用
    for n_estimators,max_depth,min_child_weight in proerty_list:
        try:
            pre_result(n_estimators, max_depth, min_child_weight,test_data)
        except:
            print('{0}-{1}-{2}eroor!'.format(n_estimators, max_depth, min_child_weight))
            continue#多组参数时应对出错