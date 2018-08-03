import numpy as np
#np.set_printoptions(threshold=np.inf)#输出数组全部
import pandas as pd
import math
from math import sqrt
from sklearn.cluster import KMeans
from pandas import Series,DataFrame
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

df= pd.read_table('u.data.txt',sep='\t',header=None)
df.columns=['client_id','Jew_id','point','time']
df=pd.DataFrame(df)
df=df.pivot('client_id','Jew_id','point')
df=df.fillna(int(0))#0补充缺失值
data=np.array(df)
estimator = KMeans(n_clusters=4,n_init=10,max_iter=10000)
result=estimator.fit_predict(data)
data1=pd.DataFrame(data,index=np.arange(1,len(data)+1,1),columns=np.arange(1,len(data[0])+1,1))
data1['cluster_id']=result
sample0=data1[data1['cluster_id']==0]
sample0.to_excel('c0.xlsx')
#共186个用户，包括用户300
sample1=data1[data1['cluster_id']==1]
sample1.to_excel('c1.xlsx')
sample2=data1[data1['cluster_id']==2]
sample2.to_excel('c2.xlsx')
sample3=data1[data1['cluster_id']==3]
sample3.to_excel('c3.xlsx')
#a=[]
#for i in range(2,11):
 #estimator = KMeans(n_clusters=i,n_init=10,max_iter=1000)
# result=estimator.fit_predict(data)
 #a.append(silhouette_score(data, result, metric='euclidean'))
#print(a)
#data1=pd.DataFrame(data,index=np.arange(1,len(data)+1,1),columns=np.arange(1,len(data[0])+1,1))




#print(data1.head())
#centroids = estimator.cluster_centers_ #获取聚类中心
#print(centroids)
#sample0=data1[data1['cluster_id']==0]
#sample0.to_excel('c0.xlsx')
#共186个用户，包括用户300
#sample1=data1[data1['cluster_id']==1]
#sample1.to_excel('c1.xlsx')
#sample2=data1[data1['cluster_id']==2]
#sample2.to_excel('c2.xlsx')
#sample3=data1[data1['cluster_id']==3]
#sample3.to_excel('c3.xlsx')
#sample4=data1[data1['cluster_id']==4]
#sample4.to_excel('c4.xlsx')
#sample5=data1[data1['cluster_id']==5]
#sample5.to_excel('c5.xlsx')
#sample6=data1[data1['cluster_id']==6]
#sample6.to_excel('c6.xlsx')


