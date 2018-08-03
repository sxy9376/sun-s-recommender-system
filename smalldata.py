import numpy as np
#np.set_printoptions(threshold=np.inf)#输出数组全部
import pandas as pd
import math
from math import sqrt
from sklearn.cluster import KMeans
from pandas import Series,DataFrame
from numpy import linalg
import matplotlib.pyplot as plt




df= pd.read_table('u.data.txt',sep='\t',header=None)

df.columns=['client_id','Jew_id','point','time']
df=pd.DataFrame(df)
#print(df)
df=df.pivot('client_id','Jew_id','point')
df=df.fillna(int(0))#0补充缺失值
data=np.array(df)
#print(data)
data1=pd.DataFrame(data,index=np.arange(1,len(data)+1,1),columns=np.arange(1,len(data[0])+1,1))
#print(data1)
#data1.to_excel('ratings.xlsx')

pf=pd.read_table('u.item.txt',sep='|',header=None)
pf=pd.DataFrame(pf)
pf.columns=['item_id','name','time','nan','web','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
del pf['name']
del pf['time']
del pf['nan']
del pf['web']
print(pf)
#pf.to_excel('shuxing.xlsx')

