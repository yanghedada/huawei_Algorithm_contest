
import pandas as pd

#迭代读取大文件
user_app_usage=pd.read_csv('user_app_usage.csv',header=None,iterator=True)

#读取前10000行
user_app_usage_demo=user_app_usage.get_chunk(10000)

#重命名特征名称
user_app_usage_demo=user_app_usage_demo.rename(columns={0:'uid',1:'app_id',2:'duration',3:'times',4:'use_date'})

#字符串转化为标准日期格式
user_app_usage_demo['use_date']=pd.to_datetime(user_app_usage_demo.use_date)

#生成星期特征,0-6分别表示周一到周末
user_app_usage_demo['weekday']=user_app_usage_demo['use_date'].dt.weekday

#生成是否工作日特征
user_app_usage_demo['isworkday']=user_app_usage_demo.weekday.apply((lambda x:x!=5 and x!=6))

#新增两列特征后数据概览
user_app_usage_demo.head(15)

#id匹配函数
def app_id_match(app_id):
    return '/'.join(app_info[app_info[0]==app_id][1].values)

#匹配
user_app_usage_demo['app_class']=user_app_usage_demo['app_id'].apply(app_id_match)