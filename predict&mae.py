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

qf=pd.read_excel('useravr.xlsx')
avrv=qf.values

ef=pd.read_excel('每个商品的均分.xlsx')
ef=ef.values

pf=pd.read_excel('usersim.xlsx')
sim=pf.values
#print(sim)
datax=DataFrame(sim,index=np.arange(1,len(sim)+1,1),columns=np.arange(1,len(sim[0])+1,1))


#ppf=pd.read_excel('改进sim.xlsx')
#sim=ppf.values

#print(sim)
#datax=DataFrame(sim,index=np.arange(1,len(sim)+1,1),columns=np.arange(1,len(sim[0])+1,1))

#dataframe:data1:所有用户对所有电影评分（data:矩阵），datax:所有用户与其余用户相似度（438*438），dataaavr:所有用户平均分（avrv：矩阵）

#对用户100推荐:
#获取用户100相似度最高的最近邻居
data300=datax[500]##########!!!!!!!!!!!!!
#print(data300)
neibor=data300.sort_values(ascending=False)
#print(neibor)
nneibor=neibor[1:11]
print(nneibor)
nneibor1=nneibor.values
#print(nneibor1[0])


#根据10个最近邻的评分预测用户100对3148个电影评分,nneibor:与用户100最近邻的10个用户id及其与100的相似度（nneibor1：矩阵），data1：所有用户对电影的评分，dataavr：所有用户平均打分

####测试部分#####
#rij = data[12][1]#从（0.0）开始
#print(rij)
#print(nneibor.index[0])#从0开始
#print(nneibor1[0])
#print(data[nneibor.index[0]-1][1])
#print(avrv)
#print(avrv[12])
#print(data[99][14])#3
#print(data[99][20])#4
###################




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
           #numerator+=nneibor1[i]*float((data[nneibor.index[i]-1][j]-avrv[nneibor.index[i]-1]))
           numerator += nneibor1[i] * float(data[nneibor.index[i] - 1][j])
           fenmu+=abs(nneibor1[i])

     score.append(numerator / (fenmu+0.0000000001))

#predict=score+avrv[499]######################!!!!!!!!!!!!!!!!!!!!-1
predict=score
#a=max(predict)
#b=min(predict)
#predict=((predict-b)/(a-b))*5
pred=pd.DataFrame(predict,index=np.arange(1,len(predict)+1,1))
#print(pred)
preds=pred.values
#print(preds)
#cols = list(pred.columns.values)
#print(cols)
recommend=pred.sort_values(by=[0],ascending=False)
print(recommend[0:10])


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
print("MAE = ", sum(sum(juedui))/(geshu+0.0000000001))
endtime = datetime.datetime.now()
print(endtime-starttime)


########RMSE#######
#epingfang= np.zeros((1,3654)).astype('float')
#print(error[0][3653]*error[0][3653])
#for i in range (0,3654):
    #epingfang[0][i]=error[0][i]*error[0][i]
#print(epingfang)
#print(sum(sum(epingfang)))
#print("RMSE = ", sqrt(sum(sum(epingfang)) / 3654))

#######MAE###########
#juedui= np.zeros((1,3654)).astype('float')
#print(abs(error[0][0]))
#for i in range(0,3654):
    #juedui[0][i]=abs(error[0][i])
#print(juedui)
#print("MAE = ", sum(sum(juedui))/3654)
