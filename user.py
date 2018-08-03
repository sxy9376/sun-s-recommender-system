from math import sqrt
from sklearn.cluster import KMeans
from pandas import Series,DataFrame
from numpy import linalg
import matplotlib.pyplot as plt
import os
import pandas as pd

#df= pd.read_table('u.data.txt',sep='\t',header=None)
#pf=pd.read_table('u.user',sep='|',header=None)
#df.columns=['client_id','item_id','point','time']
#del df['time']
#pf.columns=['client_id','age','sex','occupation','time']
#del pf['time']
#data=pd.merge(df,pf)
#data=data.sort_values('client_id')
#data=pd.DataFrame(data)
#data.to_excel('聚类要用的.xlsx')

