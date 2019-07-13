
import pandas as pd
from pandas import Series,DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt

def showData(data,name,number):
    print(name,':',data.shape)
    #print(data[0])
    print(data[:number])
    return

def showDataHead(data,name):
    print(name,':',data.shape)
    #print(data[0])
    #print(data[:5])
    print(data.head())
    return

def showFirst(data,names):
    for i in range(data.shape[1]):
        print(data.iloc[0][i])
    return

#画出每个属性的数量，用来找到空缺值
def countNUM(operation_df,name,save,path):
    data = DataFrame(operation_df)
    data.count().plot(kind = 'bar')
    
    plt.title(name)
    if save:
        plt.savefig(path+name)
    else:
        plt.show()              #显示图片
    plt.close('all')        
    return 
#画出每个属性的与第一个属性数量的差值，用来找到空缺值
def countGapNUM(operation_df,name,save,path):
    data = DataFrame(operation_df)
    t = data.count()
    t = t - t[0]
    t.plot(kind = 'bar')
    
    plt.title(name)

    if save:
        plt.savefig(path+name)
    else:
        plt.show()
    plt.close('all')  
    return 