import pandas as pd
import numpy as np
import showdata
import progressbar
import os
import warnings
from scipy import sparse
warnings.filterwarnings("ignore")
path = "../data/" 
age_train = pd.DataFrame()
age_test = pd.DataFrame()
user_basic_info = pd.DataFrame()
user_behavior_info = pd.DataFrame()
user_app_actived = pd.DataFrame()
user_app_sparse = pd.DataFrame()
app_info = pd.DataFrame()
appusedTable = pd.DataFrame()
class2id = {}
id2class = {}
is_init = False
#读取数据
def dataInit(p):
    global age_train,age_test,user_basic_info,user_behavior_info,user_app_actived,app_info,path,appusedTable,user_app_sparse
    global is_init
    is_init = True
    print ("读取数据中")
    path = p
    age_train = pd.read_csv(path+"age_train.csv", names=['uid','age_group'])
    age_test = pd.read_csv(path+"age_test.csv", names=['uid'])

    user_app_actived = pd.read_csv(path+"user_app_actived.csv", names=['uid','appId'])
    
    user_app_usage_path = path+"temp/user_app_usage2.csv"
    if os.path.exists(user_app_usage_path) == False:
        print("处理user_app_usage.csv")
        processUserAppUsage(path,user_app_usage_path)
    appusedTable = pd.read_csv(user_app_usage_path)


    user_basic_info_path = path+"temp/user_basic_info2.csv"
    if os.path.exists(user_basic_info_path) == False:
        print("处理user_basic_info缺失值")
        process_user_basic_info(path,user_basic_info_path)
    user_basic_info = pd.read_csv(user_basic_info_path, names=['uid','gender','city','prodName','ramCapacity','ramLeftRation','romCapacity','romLeftRation','color','ct','carrier','os'])

    user_behavior_info = pd.read_csv(path+"user_behavior_info.csv", names=['uid','bootTimes','AFuncTimes','BFuncTimes','CFuncTimes','DFuncTimes','EFuncTimes','FFuncTimes','FFuncSum'])
    
    
    return
    
def process_user_app_actived(path,user_app_sparse_path):
    #读取app_info
    app_info_path = path+"temp/app_info2.csv"
    if os.path.exists(app_info_path) == False:
        print("删除多余的app")
        deleteUnUsedApp(path,app_info_path)
    app_info = pd.read_csv(app_info_path, names=['appId', 'category'])

    app_info['category'] = app_info['category'].apply(lambda x: x if type(x)==str else str(x))
    sort_temp = sorted(list(set(app_info['category']))) 
    class2id['category'+'2id'] = dict(zip(sort_temp, range(1, len(sort_temp)+1)))
    app_info['category'] = app_info['category'].apply(lambda x: class2id['category'+'2id'][x]) 
    app_set = app_info['category'] 

    print((len(user_app_actived), len(app_set)))
    table = np.zeros((len(user_app_actived), len(app_set)), dtype=np.int)

    for i in range(len(user_app_actived)):
        app_list = list(user_app_actived2.iloc[i][1].split('#'))
        print(app_list)
        app_list.apply(lambda x: class2id[c+'2id'][x])  
        for l in app_list:
            for k in app_set:
                if app_set[k] == app_list[l]:
                    table[i][k] = 1
    result = user_app_actived.concat(table)
    result.to_csv(user_app_sparse_path,index=False,sep=',',encoding="utf_8_sig")
    return
#处理数据量较大的user_app_usage.csv，结合app_info.csv简单统计得到appuseProcessed.csv作为特征
def f(x):
    s = x.value_counts()
    if len(s) == 0 :
        return np.nan
    else :
        return s.index[0]
#处理user_app_usage.csv
def processUserAppUsage(path,user_app_usage_path):
    resTable = pd.DataFrame()
    #iterator=True表示逐块读取文件
    reader = pd.read_csv(path+"user_app_usage.csv", names=['uid','appId','duration','times','use_date'], iterator=True)
    last_df = pd.DataFrame()
    #读取app_info
    app_info_path = path+"app_info.csv"
    #app_info_path = path+"temp/app_info2.csv"
    if os.path.exists(app_info_path) == False:
        print("删除多余的app")
        deleteUnUsedApp(path,app_info_path)
    app_info = pd.read_csv(app_info_path, names=['appId', 'category'])
    
    #将app_info['category']中的分类形成集合，再变为列表
    cats = list(set(app_info['category']))

    #形成字典
    category2id = dict(zip(sorted(cats), range(0,len(cats))))
    #将 app_info['category']原来中文的分类，改为上面形成的字典中的值
    app_info['category'] = app_info['category'].apply(lambda x: category2id[x])

    i = 1
    while True:
        try:
            #输出index
            print("index: {}".format(i))
            i+=1
            df = reader.get_chunk(1000000)
            #数据拼接
            df = pd.concat([last_df, df])
            #shape[0] 矩阵长度
            idx = df.shape[0]-1
            #iat[idx,0]，第idx行，第0列
            last_user = df.iat[idx,0]
            while(df.iat[idx,0]==last_user):
                idx-=1
            last_df = df[idx+1:]
            df = df[:idx+1]

            now_df = pd.DataFrame()
            now_df['uid'] = df['uid'].unique()
            df_groupby_uid = df.groupby('uid')
            now_df = now_df.merge(df_groupby_uid['appId'].count().to_frame(), how='left', on='uid')
            now_df = now_df.merge(df_groupby_uid['appId','use_date'].agg(['nunique']), how='left', on='uid')
            now_df = now_df.merge(df_groupby_uid['duration','times'].agg(['mean','max','std']), how='left', on='uid')    

            

            df = df.merge(app_info, how='left', on='appId')
            now_df.columns = ['uid','usage_cnt','usage_appid_cnt','usage_date_cnt','duration_mean','duration_max','duration_std','times_mean','times_max','duration_std']
            df_groupby_uid = df.groupby('uid')
        
            now_df = now_df.merge(df_groupby_uid['category'].nunique().to_frame(), how='left', on='uid')
            now_df['most_time_category'] = df.groupby(['uid'])['category'].transform(f)

            #print(df.groupby(['uid'])['category'].value_counts().index[0])
            #print(df.iloc[df_groupby_uid['duration'].idxmax()].head())
            now_df = now_df.merge(df.iloc[ df_groupby_uid['duration'].idxmax()],how='left', on='uid')
            now_df.drop('appId',axis=1, inplace=True)
            now_df.drop('use_date',axis=1, inplace=True)
            now_df.rename(columns={'category':'most_duration_category'}, inplace = True)
            now_df.drop('duration',axis=1, inplace=True)
            
            
            
            resTable = pd.concat([resTable, now_df])

        except StopIteration:
            break
    
    resTable.to_csv(user_app_usage_path,index=0)
    
    print("处理user_app_usage.csv完成")
    return

#创建需要的文件夹
def  init_dir(path):
    dirP = [path+"img",path+"temp",path+"try"]
    for d in dirP:
        if os.path.exists(d) == False:
            os.mkdir(d)

#处理user_basic_info缺失值
def  process_user_basic_info(path,user_basic_info_path):
    user_basic_info = pd.read_csv(path+"user_basic_info.csv", names=['uid','gender','city','prodName','ramCapacity','ramLeftRation','romCapacity','romLeftRation','color','fontSize','ct','carrier','os'])
    user_basic_info.drop(['fontSize'],axis=1,inplace=True)
    user_basic_info['city'] = user_basic_info['city'].fillna("other")
    user_basic_info['ct'] = user_basic_info['city'].fillna("other")
    mean_replace = ['ramCapacity','ramLeftRation','romCapacity','romLeftRation']
    for t in mean_replace:
        user_basic_info[t] = user_basic_info[t].fillna(user_basic_info[t].mean())
    user_basic_info.to_csv(user_basic_info_path,index=False,sep=',',encoding="utf_8_sig",header=0)
    return

#删除未使用的app
def deleteUnUsedApp(path,app_info_path):
    user_app = user_app_actived['appId']
    user_app_list = list()

    temp_path1 = path+"temp/deleteUnUsedApp1.csv"
    temp_path2 = path+"temp/deleteUnUsedApp2.csv"
    if os.path.exists(temp_path1):
        user_app_list = readCSV2List(temp_path1)
        del user_app_list[0]
    else:
        p = progressbar.ProgressBar(user_app_actived.size)
        p.start()
        for i in range(user_app.size-1):
            p.update(i)
            user_app_list.append(set(user_app[i].split('#')))
        p.finish()
        list_file = pd.DataFrame({'list_file':user_app_list})
        list_file.to_csv(temp_path1,index=False,sep=',')


    if os.path.exists(temp_path2):
        appSet = readCSV2List(temp_path2)
        del appSet[0]
    else:
        p = progressbar.ProgressBar(user_app_actived.size)
        p.start()
        appSet = user_app_list[0]

        for i in range(user_app.size-1):
            #交集
            p.update(i)
            
            appSet = appSet|set(user_app_list[i])
        
        p.finish()  
        appSetFile = pd.DataFrame({'appId':list(appSet)})
        appSetFile.to_csv(temp_path2,index=False,sep=',')

    appSetFile = pd.read_csv(temp_path2)
    app_info = pd.read_csv(path+"app_info.csv", names=['appId', 'category'])

    resTable = appSetFile.merge(app_info, how='left', on='appId')
    resTable.fillna(" ",inplace = True)
    resTable.to_csv(app_info_path,index=False,sep=',',encoding="utf_8_sig",header=0)
    
    return


def getTest():
    return getData(age_test)
def getTrain():
    return getData(age_train)
def getData(age_data):
    resTable = age_data.merge(user_basic_info, how='left', on='uid')
    resTable = resTable.merge(user_behavior_info, how='left', on='uid')
    resTable = resTable.merge(appusedTable, how='left', on='uid')
    resTable = dictSomeAttributes(resTable,['city','prodName','color','carrier','ct','carrier'])
    resTable = resTable.merge(user_app_actived, how='left', on='uid')
    resTable['appId'] = resTable['appId'].apply(lambda x: len(list(x.split('#'))))

    resTable = resTable.fillna(0)
    return resTable


#字典化处理
def dictSomeAttributes(table,cat_columns):
    for c in cat_columns:
        #将内容转为字符串
        table[c] = table[c].apply(lambda x: x if type(x)==str else str(x))
        #将一列的内容转为集合，在排序
        sort_temp = sorted(list(set(table[c]))) 
        #将 集合 字典化[1,2,3,,,,len(sort_temp)]
        class2id[c+'2id'] = dict(zip(sort_temp, range(1, len(sort_temp)+1)))
        #将 [1,2,3,,,,len(sort_temp)] 字典化
        id2class['id2'+c] = dict(zip(range(1,len(sort_temp)+1), sort_temp))
        #将table字典化
        table[c] = table[c].apply(lambda x: class2id[c+'2id'][x])    
    return table
#处理user_basic_info、user_behavior_info表格




#寻找缺失值
def find_issing_value(path):

    if is_init == False:
        dataInit(path)
    list = [user_basic_info,user_behavior_info,user_app_actived,app_info]
    name = ["user_basic_info","user_behavior_info","user_app_actived","app_info"]
    for i in range(4):
        showdata.countNUM(list[i],name[i],True,path+"img/")
        showdata.countGapNUM(list[i],name[i]+"_gap",True,path+"img/")

#读取CSV为List
def readCSV2List(filePath):
    try:
        file=open(filePath,'r',encoding="gbk")# 读取以utf-8
        context = file.read() # 读取成str
        list_result=context.split("\n")#  以回车符\n分割成单独的行
        #每一行的各个元素是以【,】分割的，因此可以
        length=len(list_result)
        for i in range(length):
            list_result[i]=list_result[i].split(",")
        list_result
        return list_result
    except Exception :
        print("文件读取转换失败，请检查文件路径及文件编码是否正确")
    finally:
        file.close();# 操作完成一定要关闭