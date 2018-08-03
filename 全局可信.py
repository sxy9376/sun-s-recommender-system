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


count=pd.read_excel('active.xlsx',header=None)
count=np.array(count)
#print(count[0:10])
#print(count[0])

mean=pd.read_excel('每个商品的均分.xlsx',header=None)
mean=np.array(mean)
#print(mean[0:10])
#print(mean[0])

avr=pd.read_excel('useravr.xlsx',header=None)
avr=np.array(avr)


equ= np.zeros((1,943)).astype('float')
for i in range(0,943):
    fenzi=0
    for j in range(0,1682):
        if data[i][j]!=0:
         fenzi+=(data[i][j]-avr[i])*(data[i][j]-avr[i])
    equ[0][i]=math.sqrt(fenzi/count[i])
#print(equ[0][0])
equ=pd.DataFrame(equ)
equ.to_excel('equ.xlsx')


cor= np.zeros((1,943)).astype('float')
for i in range(0,943):
    fenmu=0
    for j in range(0,1682):
        if data[i][j] != 0:
            fenmu+=abs(data[i][j]-mean[j])
    cor[0][i]=1/fenmu
#print(cor[0][0])
cor=pd.DataFrame(cor)
cor.to_excel('cor.xlsx')