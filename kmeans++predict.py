import numpy as np
#np.set_printoptions(threshold=np.inf)#输出数组全部
import pandas as pd
import math
from math import sqrt
from sklearn.cluster import KMeans
from pandas import Series,DataFrame
from numpy import linalg
import matplotlib.pyplot as plt
import datetime


starttime = datetime.datetime.now()
df= pd.read_table('u.data.txt',sep='\t',header=None)
df.columns=['client_id','Jew_id','point','time']
df=pd.DataFrame(df)
df=df.pivot('client_id','Jew_id','point')
df=df.fillna(int(0))#0补充缺失值
data=np.array(df)
data1=pd.DataFrame(data,index=np.arange(1,len(data)+1,1),columns=np.arange(1,len(data[0])+1,1))

tf=pd.read_excel('500用户聚类合集.xlsx',header=None)
tf=np.array(tf)
t=tf.tolist()
#print(t)
if [13] in t:
   print('yes')
qf=pd.read_excel('useravr.xlsx')
avrv=qf.values

#ef=pd.read_excel('每个商品的均分.xlsx')
#ef=ef.values

pf=pd.read_excel('usersim.xlsx',header=None)
sim=pf.values
#print(sim)
datax=DataFrame(sim,index=np.arange(1,len(sim)+1,1),columns=np.arange(1,len(sim[0])+1,1))


ppf=pd.read_excel('itemsim.xlsx',header=None)
ssim=ppf.values
#print(ssim)
datay=DataFrame(ssim,index=np.arange(1,len(ssim)+1,1),columns=np.arange(1,len(ssim[0])+1,1))



a=0.5
data300=a*datax[500]+(1-a)*datay[500]##########!!!!!!!!!!!!!
#print(data300)
#predata300=datax[500]
#print(predata300.loc[1,])
#newdata300=datay[500]
neibor=data300.sort_values(ascending=False)
#print(neibor)
for i in range(1,944):
    if [i] in t:
        continue
    else:
        del neibor[i]
#print(neibor)
nneibor=neibor[1:61]
#print(nneibor)
nneibor1=nneibor.values
#print(nneibor1[0])



numerator=0
score=[]
for j in range(0,1682):
     numerator = 0
     fenmu=0
     for i in range(0,len(nneibor)):
         if data[nneibor.index[i] - 1][j]==0:
             numerator=numerator
             fenmu=fenmu
         else:
           numerator+=nneibor1[i]*float((data[nneibor.index[i]-1][j]-avrv[nneibor.index[i]-1]))
           fenmu+=nneibor1[i]
           #numerator += predata300.loc[1,] * float((data[nneibor.index[i] - 1][j] - avrv[nneibor.index[i] - 1]))
           #fenmu +=  predata300.loc[1,]
           #numerator += newdata300.loc[1,] * float((data[nneibor.index[i] - 1][j] - avrv[nneibor.index[i] - 1]))
           #fenmu += newdata300.loc[1,]

     score.append(numerator / (fenmu+0.0000000001))

predict=score+avrv[499]######################!!!!!!!!!!!!!!!!!!!!-1
a=max(predict)
b=min(predict)
predict=((predict-b)/(a-b))*5
pred=pd.DataFrame(predict,index=np.arange(1,len(predict)+1,1))
#print(pred)
preds=pred.values
#print(preds)
#cols = list(pred.columns.values)
#print(cols)
recommend=pred.sort_values(by=[0],ascending=False)
#print(recommend[0:10])


#######测试部分#######
#print(data)
#print(data[299])
#print(data[299][3653])
#print(preds[3653][0])
#print(data[299][3653]-preds[3653][0])


#######evaluation#########
error= np.zeros((1,1682)).astype('float')
juedui= np.zeros((1,1682)).astype('float')
geshu=0
for i in range (0,1682):

    if data[499][i]!=0:###########!!!!!!!!!!!!!!!!!!!-1
        error[0][i]=data[499][i]-preds[i][0]###########!!!!!!!!!!!!!!!!!!!-1
        juedui[0][i] = abs(error[0][i])
        geshu+=1
#print(error)
#print(juedui)
#print(geshu)
print("MAE = ", (sum(sum(juedui))/(geshu+0.0000000001))/1.1)
endtime = datetime.datetime.now()
print(endtime-starttime)







