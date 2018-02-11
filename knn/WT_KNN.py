#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:37:56 2017

@author: rajathanda
"""

import pandas as pd
import operator
import sys
from sklearn.metrics import mean_squared_error as mse
from scipy.spatial import distance
from math import sqrt
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

def pd_to_list(df):
    temp=[]
    for row in df.iterrows():
        index, data = row
        temp.append(data.tolist())
    return temp

def User_Filt(Train,Movie):
    Filt_df = Train[Train[Movie] > 0]
    return Filt_df

#-----User Matrix--------
Ratings_df = df_train.pivot(index = 'UserID', columns ='MovieID', values = 'Ratings').fillna(0)

#----- Neighbors Calculation-------

a=[]
def NeighborsCompute(Train,Test,k):
    T_user=int(Test[0])
    T_Movie=Test[1]
    T_Ratings=Test[2]
    Filt_Users_df=Train[Train[T_Movie] > 0]
    Filt_Users_Index=Filt_Users_df.index.values
    testUser=Train[Train.index == T_user]
    Euc_Dist=[]
    for users in Filt_Users_Index:
        Dist = distance.euclidean(Filt_Users_df[Filt_Users_df.index == users],testUser)
        Euc_Dist.append([T_user,users,int(Filt_Users_df[Filt_Users_df.index == users][T_Movie].values),Dist])
    a=Euc_Dist.sort(key=operator.itemgetter(3))
    Track = len(Euc_Dist)
    neighbors = []
    [[neighbors.append(Euc_Dist[n]) for n in range(k)] if (Track > k) else [neighbors.append(Euc_Dist[n]) for n in range(Track)]
        ]
    
    return neighbors

#---Weighted KNN-----------
def Weight_pred(neigh):
    W_Dist=[]
    W_Ratings=[]
    for x in range(len(neigh)):
        W_Dist.append(neigh[x][3])
        W_Ratings.append(neigh[x][2])
    Far_NN=max(W_Dist)
    Near_NN=min(W_Dist)
    Wi=[]
    Di=[]
    In_preds=0
    Di_ratings=[]
    if(Far_NN-Near_NN!=0):
        for x in range(len(neigh)):
            Wi.append((Far_NN-neigh[x][3])/(Far_NN-Near_NN))
            Di.append(Wi[x]*W_Dist[x])
            Di_ratings.append(Di[x]*W_Ratings[x])
        preds=round(sum(Di_ratings)/sum(Di),1)
    else:
        for x in range(len(neigh)):
            In_preds += neigh[x][2]
        preds = round(In_preds/float(len(neigh)), 1)    
    return(preds)  
    
    

final= []
preds=[]
start = timeit.default_timer()
Ts_List=pd_to_list(df_test)
test_users = int(len(Ts_List))
k = 3
for user in range(test_users):
    neigh = NeighborsCompute(Ratings_df, Ts_List[user], k)
    ratings = UW_pred(neigh)
    preds.append([Ts_List[user][0],Ts_List[user][1],Ts_List[user][2],ratings])
    final.append(ratings)
stop = timeit.default_timer()
stop = stop - start
print(stop)
pred = pd.DataFrame(preds)
pred.to_csv(outFileName)

#-RMSE Calculation------
columns=["UserID","MovieID","Ratings","Predicitons"]
pred=pd.read_csv(outFileName,names=columns)
rms = sqrt(mse(pred["Ratings"],pred["Predicitons"]))

#-------END--------------


