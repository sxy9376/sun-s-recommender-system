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

datavr=pd.read_excel('useravr.xlsx',header=None)
#print(datavr[0:10])
avr=datavr.values
#print(len(avr))
#print(avr[0])

sim= np.zeros((943,943)).astype('float')
#print(sim[0][0])
for p in range(0,943):
 for j in range(0,943):
  a1=[]
  a2=[]
  for i in range(0,1682):
     if  int(data[p][i])&int(data[j][i])!=0:
       a1.append(data[p][i])
       a2.append(data[j][i])

 #print(a1)
 #print(a2)
 #print(a1[1])
 #print(len(a1))
 #s1=len(a1)
 #if s1!=0:
  #print(a1[s1-1])

  fenzi=0
  m1=0
  m2=0
  m3=0
 #point=0
  if len(a1)==0:
      sim[p][j]=0
  else:
   for k in range(0,len(a1)):
     fenzi+=(a1[k]-avr[p])*(a2[k]-avr[j])
     m1+=(a1[k]-avr[p])*(a1[k]-avr[p])
     m2+=(a2[k]-avr[j])*(a2[k]-avr[j])
   m1=math.sqrt(m1)
   m2=math.sqrt(m2)
   m3=(m1*m2)+0.000000001#分母有0的情况
 #point=fenzi/m3
   sim[p][j]=fenzi/m3
#print(sim)
sim=pd.DataFrame(sim)
sim.to_excel('usersim.xlsx')






