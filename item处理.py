import numpy as np
import pandas as pd
import math
from math import sqrt
from sklearn.cluster import KMeans
from pandas import Series,DataFrame
from numpy import linalg
import matplotlib.pyplot as plt
import os


#df= pd.read_table('u.data.txt',sep='\t',header=None)
#pf=pd.read_excel('shuxing.xlsx',header=None)
#df.columns=['client_id','item_id','point','time']
#del df['time']
#pf.columns=['item_id','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
#data=pd.merge(df,pf)
#data=data.sort_values('client_id')
#data.to_excel('final.xlsx')

df= pd.read_excel('final.xlsx',sep='\t',header=None)
df.columns=['client_id','item_id','point','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
del df['item_id']
del df['point']
m=df['client_id']  #选择表格中的''client_id'列，返回的是DataFrame类型
data=np.array(m)

####Nu用户评价过的电影个数######
n= np.zeros((1,943)).astype('int64')
for i in range(1,944):

  n[0][i-1]=data[data==i].size
#print(n[0][0])


#####用户评价过的Gi类别的电影数#######
pp= np.zeros((943,19)).astype('int64')
for j in range(1,944):
    p = df[df['client_id'] == j]
    for i in range (1,20):   #属性1对应i=2
        pp[j-1][i-1]=p.iloc[:,i].sum()

#print(pp[i][j])
#pp=pd.DataFrame(pp)
#pp.to_excel('评价过某一类别的用户总数.xlsx')

######评价过GI类别电影的用户数######
gg=[10,938,901,659,805,940,914,352,943,512,618,789,754,897,943,908,937,925,491]
#print(gg[0])

######943*19的特征偏好矩阵#######
zz=np.zeros((943,19)).astype('float')
for i in range(0,943):
    for j in range(0,19):
        zz[i][j]=(pp[i][j]/n[0][i])*math.log(943/gg[j])
zz=pd.DataFrame(zz)
#zz.to_excel('特征偏好矩阵.xlsx')



tf=pd.read_excel('用户在特征中的均分.xlsx',header=None)
tf=np.array(tf)
ef=pd.read_excel('特征偏好矩阵.xlsx',header=None)
ef=np.array(ef)
#print(tf[0])
#print(ef[1][0])
####计算特征值####
sim= np.zeros((943,943)).astype('float')
#print(sim[0][0])
for p in range(0,943):
 for j in range(0,943):
  a1=[]
  a2=[]
  for i in range(0,19):
     if  (ef[p][i]!=float(0))&(ef[j][i]!=float(0)):
       a1.append(ef[p][i])
       a2.append(ef[j][i])


  fenzi=0
  m1=0
  m2=0
  m3=0

  if len(a1)==0:
      sim[p][j]=0
  else:
   for k in range(0,len(a1)):
     fenzi+=(a1[k]-tf[p])*(a2[k]-tf[j])
     m1+=(a1[k]-tf[p])*(a1[k]-tf[p])
     m2+=(a2[k]-tf[j])*(a2[k]-tf[j])
   m1=math.sqrt(m1)
   m2=math.sqrt(m2)
   m3=(m1*m2)+0.000000001#分母有0的情况
 #point=fenzi/m3
   sim[p][j]=fenzi/m3
#print(sim)
sim=pd.DataFrame(sim)
sim.to_excel('itemsim.xlsx')