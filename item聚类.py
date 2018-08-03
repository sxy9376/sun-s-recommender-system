from __future__ import print_function
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd




df= pd.read_excel('特征偏好矩阵.xlsx',header=None)
df=pd.DataFrame(df)
data=np.array(df)
#a=[]
#for i in range(2,11):
estimator = KMeans(n_clusters=4,n_init=10,max_iter=10000)
result=estimator.fit_predict(data)
#a.append(silhouette_score(data, result, metric='euclidean'))
#print(a)
data1=pd.DataFrame(data,index=np.arange(1,len(data)+1,1),columns=np.arange(1,len(data[0])+1,1))
data1['cluster_id']=result
sample0=data1[data1['cluster_id']==0]
sample0.to_excel('0.xlsx')
#共186个用户，包括用户300
sample1=data1[data1['cluster_id']==1]
sample1.to_excel('1.xlsx')
sample2=data1[data1['cluster_id']==2]
sample2.to_excel('2.xlsx')
sample3=data1[data1['cluster_id']==3]
sample3.to_excel('3.xlsx')
