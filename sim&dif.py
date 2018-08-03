import numpy as np
#np.set_printoptions(threshold=np.inf)#输出数组全部
import pandas as pd
import math
from math import sqrt
from sklearn.cluster import KMeans
from pandas import Series,DataFrame
from numpy import linalg
import matplotlib.pyplot as plt

df=pd.read_excel('usersim.xlsx')
pf=pd.read_excel('dif.xlsx')
df=df.values
pf=pf.values

gaijin= np.zeros((943,943)).astype('float')
for i in range(0,943):
    for j in range(0,943):
        gaijin[i][j]=df[i][j]*pf[i][j]
gaijin=pd.DataFrame(gaijin)
gaijin.to_excel('改进sim.xlsx')