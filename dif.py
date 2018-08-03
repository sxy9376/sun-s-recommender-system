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




dif= np.zeros((943,943)).astype('float')
for p in range(0,943):
 for j in range(0,943):
  a1=[]
  a2=[]
  a3=[]
  for i in range(0,1682):
     if  int(data[p][i])&int(data[j][i])!=0:
       a1.append(data[p][i])
       a2.append(data[j][i])
 #print(a1)
 #print(a2)
  s1=len(a1)
  if s1==0:
      dif[p][j]=float('inf')
  else:
   for k in range(0,s1):
       a3.append(a1[k]-a2[k])
       a3[k]=a3[k]*a3[k]
  #print(a3)
   dif[p][j]=math.exp(-(math.sqrt(sum(a3)))/s1)
dif=pd.DataFrame(dif)
print(dif)
#dif.to_excel('dif.xlsx')









