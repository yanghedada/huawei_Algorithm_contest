'''
调节参数max_depth、min_child_weight
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

'''
第一轮参数调整得到的n_estimators最优值（1200），其余参数继续默认值
用交叉验证评价模型性能时，用scoring参数定义评价指标。
评价指标是越高越好，因此用一些损失函数当评价指标时，需要再加负号，
如neg_log_loss，neg_mean_squared_error 
'''

#max_depth 建议3-10， min_child_weight=1／sqrt(ratio_rare_event) =8
max_depth = [6,8,10]
min_child_weight = [6,8,10]#形成参数网格
param_test2_2 = dict(max_depth=max_depth, min_child_weight=min_child_weight)#构建参数字典

xgb2_2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1200,  #第一轮参数调整得到的n_estimators最优值
        max_depth=5,
        min_child_weight=1,#这两个值在这里设置不影响，网格搜索会重新设置
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel = 0.7,
        objective= 'multi:softprob',
        n_jobs=8,#多线程跑模型
        gpu_id=0,#如果有GPU，GPU序号
        tree_method='gpu_hist',#采用GPU
        predictor='cpu_predictor'),#防止GPU内存不足，采用CPU预测
        seed=3)

if __name__=='__main__':
    gsearch2_2 = GridSearchCV(xgb2_2, param_grid = param_test2_2, scoring='neg_log_loss',n_jobs=-1, cv=kfold)#网格搜索
    gsearch2_2.fit(X_train , y_train)
    print('gsearch2_2.grid_scores_:{0}\ngsearch2_2.best_params_:{1}\n,gsearch2_2.best_score_:{2}\n'
    .format(gsearch2_2.grid_scores_,gsearch2_2.best_params_,gsearch2_2.best_score_))#打印网格搜索结果
    gsearch2_2.cv_results_
    #打印不同参数模型结果
    print("Best: %f using %s" % (gsearch1.best_score_, gsearch1.best_params_))
    test_means  = gsearch2_2.cv_results_['mean_test_score']
    test_stds   = gsearch2_2.cv_results_['std_test_score']
    train_means = gsearch2_2.cv_results_['mean_train_score']
    train_stds  = gsearch2_2.cv_results_['std_train_score']

    pd.DataFrame(gsearch2_2.cv_results_).to_csv('my_preds_maxdepth_min_child_weights.csv')

    # 画出不同参数下损失曲线
    test_scores  = np.array(test_means).reshape(len(min_child_weight), len(max_depth))
    train_scores = np.array(train_means).reshape(len(min_child_weight), len(max_depth))

    for i, value in enumerate(min_child_weight):
        pyplot.plot(max_depth, test_scores[i], label= 'test_min_child_weight:'   + str(value))
    pyplot.legend()
    pyplot.xlabel('max_depth')
    pyplot.ylabel('- Log Loss')
    pyplot.savefig('max_depth_vs_min_child_weght.png')

'''
粗略调节max_depth以及min_child_weight之后缩小步长继续调参，
如：由上面的步长为2调节为步长为1，继续网格搜索
代码同上，改动部分位置即可
'''