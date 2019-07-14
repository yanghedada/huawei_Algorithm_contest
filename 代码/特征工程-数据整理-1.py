'''
处理app_info表的信息，将app编号转化为app类别
并分类记录数量
'''

#导入相关模块
import pandas as pd
import seaborn as sns

user_app=pd.read_csv('app_info.csv',header=None)#读取app安装记录表

#步骤一：写入app_info
app_info=pd.read_csv('app_info.csv',header=None)

#步骤二:重写特征名
app_info=app_info.rename(columns={0:'app_id',1:'app_class'})

#步骤三:建立类别列表
app_class_list=list(app_info.app_class.unique())

#步骤四:写入user_app_actived.csv
user_app=pd.read_csv('user_app_actived.csv',header=None)

#步骤五:重写特征名
user_app=user_app.rename(columns={0:'user_id',1:'app_list'})

#步骤六:加入类别特征，将初始次数赋值为0
for app in app_class_list:
    user_app[app]=pd.Series(data=[0]*2512500)

#步骤七:提取app列
user_app_app_list=user_app[['user_id','app_list']]

#步骤八:提取类型列
user_app_app_class=user_app.iloc[:,2:]

#计数
for i in range(0,2512500):
    #分割app字符串
    user_apps=user_app_app_list.iloc[i,1].split('#')
    for app in user_apps:
        #匹配app_id和app类别
        app_values=app_info[app_info.app_id==app].app_class.values
        #对于每一个用户ID统计所使用的各种app的类别数
        for app_value in app_values:
            user_app_app_class.iloc[i,:][app_value]+=1
    #每处理10个用户打印一次，方便观察进度
    if i%10==0:
        print('deal-{}'.format(i))
'''
由于数据超大，所以按以上代码分别处理成几块数据后，以下代码为合并文件
'''
data1=pd.read_csv('./app_class/data-100000.csv',index_col=0)
file_list=list(range(200000,2300000,100000))
+list(range(2220000,2320000,20000))
+list(range(2400000,2520000,20000))+[2510000]#文件序号

data=pd.read_csv('./app_class/data-100000.csv',index_col=0)

for i in file_list:
    file_path='./app_class/data-'+str(i)+'.csv'#合并数据
    data=pd.concat([data,pd.read_csv(file_path,index_col=0)],axis=0)