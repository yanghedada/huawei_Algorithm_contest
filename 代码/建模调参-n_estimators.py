'''
调节参数n_estimators
'''
#import导入必要的模块
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
'''
from xgboost import XGBClassifier
import xgboost as xgb

import pandas as pd 
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from matplotlib import pyplot
import seaborn as sns
%matplotlib inline

#读取数据
train_data=pd.read_csv('train_new.csv',index_col=0)#训练数据
train_data.head()

#训练集处理
X=train_data.drop(['target_age','uid'],axis=1)#去除训练数据中的目标变量和id变量
Y=train_data.target_age#提取训练数据中的目标变量
X_train = np.array(X)#构建特征训练矩阵

#观察预测目标分布，若不平衡考虑分层抽样
sns.countplot(Y)
pyplot.xlabel('target_age')
pyplot.ylabel('Number of occurrences')

#构建K折交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

#构建n_estimators选择函数，观察图像选择收敛的n_estimators(树数目)

def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=None, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 6#预测目标的类别数
        
        xgtrain = xgb.DMatrix(X_train, label = y_train)#构建xgb格式的数据集
        
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds =cv_folds,
             metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        
        n_estimators = cvresult.shape[0]
        alg.set_params(n_estimators = n_estimators)
        
        print (cvresult)
        cvresult.to_csv('my_preds.csv', index_label = 'n_estimators')#保存网格搜索结果
        
        # 画出n_estimators与log损失的关系图
        test_means = cvresult['test-mlogloss-mean']
        test_stds = cvresult['test-mlogloss-std'] 
        
        train_means = cvresult['train-mlogloss-mean']
        train_stds = cvresult['train-mlogloss-std'] 

        x_axis = range(0, n_estimators)
        pyplot.errorbar(x_axis, test_means, yerr=test_stds ,label='Test')
        pyplot.errorbar(x_axis, train_means, yerr=train_stds ,label='Train')
        pyplot.title("XGBoost n_estimators vs Log Loss")
        pyplot.xlabel( 'n_estimators' )
        pyplot.ylabel( 'Log Loss' )
        pyplot.savefig( 'n_estimators.png' )
    
    #用选择参数训练训练集
    alg.fit(X_train, y_train, eval_metric='mlogloss')
        
    #训练集损失
    train_predprob = alg.predict_proba(X_train)
    logloss = log_loss(y_train, train_predprob)

    print ("logloss of train :" )
    print (logloss)

if __name__=='__main__':
#其他参数固定，观察选择n_estimators
    xgb = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,  #数值大没关系，cv会自动返回合适的n_estimators
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel=0.7,
        objective= 'multi:softprob',
        n_jobs=8,#多线程跑模型
        gpu_id=0,#如果有GPU，GPU序号
        tree_method='gpu_hist',#采用GPU
        predictor='cpu_predictor'),#防止GPU内存不足，采用CPU预测
        seed=3)
    modelfit(xgb, X_train, y_train, cv_folds = kfold)