import pandas as pd
import numpy as np
import lightgbm as lgb
import showdata
import moudles
import processData

path = "../data/"
#数据处理
processData.dataInit(path)
df_train = processData.getTrain()
df_test = processData.getTest()

#训练模型
df = moudles.useModel(df_train,df_test)

#保存文件
df.to_csv(path+'submission.csv',index=False)
