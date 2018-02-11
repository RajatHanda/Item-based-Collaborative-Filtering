#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 03:57:19 2017

@author: rajathanda
"""

import pandas as pd
import numpy as np
import sys
from math import sqrt
from datetime import datetime
import timeit


USERIDX=0
MOVIEIDX=1
RATINGIDX=2

if len(sys.argv) != 3:
	print('usage: trainingDataFileName testDataFileName predictionsFile')
	sys.exit(99)

trainingFileName = sys.argv[1]
testFileName     = sys.argv[2]
outFileName      = sys.argv[3]


columns=["UserID","MovieID","Ratings","Timestamp"]
df_train=pd.read_csv(trainingFileName,names=columns)
df_test=pd.read_csv(testFileName,names=columns)

#--Normalizing Data----------
all_User=list(df_train['UserID'].unique())
avg_rating = []
for i in range(len(all_User)):
    df_avg = df_train.loc[df_train.UserID==i]
    avg_rating.append([i,np.average(df_avg.Ratings.values)])
df_train['Avg']=0
for i in range(len(all_User)):
    df_train['Avg'][df_train.UserID == i] = avg_rating[i][1]

df_train['norm_rating'] = df_train.Ratings - df_train.Avg
df_matrix = df_train.pivot(index='UserID', columns='MovieID', values='norm_rating').fillna(0)

#----Simillarity Metric------------
def cosine_cf(vec1,vec2):
    num = 0
    num1 = 0
    num2 = 0
    for i in range(len(vec1)):
        num += vec1[i]*vec2[i]
        if(vec1[i]==0 or vec2[i]==0):
            continue
        else:
            num1 += vec1[i]**2
            num2 += vec2[i]**2
    tot = sqrt(num1) * sqrt(num2)
    simillarity = num/tot
    return simillarity

s_mat = pd.DataFrame(np.zeros((9216, 9216)))
s_mat = s_mat.replace(0,-2)

#-------Matric Computation----------
def sim_compute(test,df_org,df_filter):
    sim=[]
    Test_movie=int(test.MovieID.values)
    T_user=int(test.UserID.values)
    Rated_Movies=df_org.loc[df_org.UserID==T_user].MovieID.values
    for r in Rated_Movies:
        if s_mat[Test_movie][r]==-2:
            sim_f=cosine_cf(df_filter[Test_movie],df_filter[r])
            s_mat[Test_movie][r]=sim_f
            s_mat[r][Test_movie]=sim_f
            Rat=df_filter[r][T_user]
            sim.append((r,sim_f,T_user,Test_movie,Rat))
        else:
            sim_f=s_mat[Test_movie][r]
            Rat=df_filter[r][T_user]
            sim.append((r,sim_f,T_user,Test_movie,Rat))
    return sim

#---Making Predicitons---------------
start = timeit.default_timer()
pred=[]
count=0
for rec in range(len(df_test)):
    count+=1
    R_Sim=sim_compute(df_test.loc[rec:rec,],df_train,df_matrix)
    tmp1=0
    tmp2=0
    for i in R_Sim:
        Mov=int(i[0])
        M_sim=i[1]
        Use=int(i[2])
        pred_M=i[3]
        if(M_sim>0):
            tmp3=df_train[(df_train.MovieID==Mov)&(df_train.UserID==Use)].Ratings.values*M_sim
            tmp2+=tmp3[0]
            tmp1+=M_sim
            tmp4=df_test[(df_test.MovieID==pred_M)&(df_test.UserID==Use)].Ratings.values
    pred.append([Use,pred_M,tmp4,tmp2,tmp1])
    if(count%100 == 0):
        print('Step:',count, 'time:',str(datetime.now().time()))
stop = timeit.default_timer()
print('total time:',stop - start)

col = ['UserID','MovieID','Ratings','sim1','sim2']
df_preds= pd.DataFrame(pred, columns=col)
df_preds['prediction'] = round(df_preds.sim1/df_preds.sim2,1)
original_rating = []
for i in range(len(df_preds.Ratings)):
    original_rating.append(df_preds.Ratings[i][0])
df_preds = df_preds.drop('Ratings', 1)
df_preds = df_preds.drop('sim1', 1)
df_preds = df_preds.drop('sim2', 1)
df_preds['Ratings'] = original_rating

df_preds = df_preds[['UserID','MovieID','Ratings','prediction']]
df_preds.to_csv(outFileName)

#----------------MSE calculation-------------
mse20 = (((df_preds.prediction - df_preds.Ratings) ** 2).sum()) / len(df_preds.prediction)
pow(mse20,0.5) 
print(mse20)

#-------------------END----------------------










