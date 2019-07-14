'''
尝试组合的特征：

商务、实用工具、便捷生活、出行导航、购物比价
动作射击、动作冒险
教育、学习办公、图书阅读
休闲益智、棋牌桌游、休闲游戏、棋牌天地、益智棋牌、经营策略
app总个数
'''
#导入相关模块
'''
stats:统计模块
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#读取数据
train=pd.read_csv('new_train_data_2.csv',index_col=0)
test=pd.read_csv('new_test_data_2.csv',index_col=0)

#组合特征构建函数
def new_fea(data,num):#num表示开始的特征序号，即colnums=？
    def adds(*args):
        return sum(args)
    businus=adds(data['商务'],data['实用工具'],data['便捷生活'],data['出行导航'],data['购物比价'])
    act=adds(data['动作射击'],data['动作冒险'])
    learn=adds(data['教育'],data['学习办公'],data['图书阅读'])
    relax=adds(data['休闲益智'],data['棋牌桌游'],data['休闲游戏'],data['棋牌天地'],data['益智棋牌'],data['经营策略'])
    appsum=data.iloc[:,num:].sum(axis=1)
    for name in ['businus','act','learn','relax','appsum']:
        data[name]=locals()[name]
    return data

new_test_data=new_fea(train,22)
new_test_data=new_fea(test,22)