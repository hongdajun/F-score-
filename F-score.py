# coding=gbk
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
df_jiaceng= pd.read_csv('sjzscore.csv',header = None,sep=',')
df_jiaceng.columns = ['Class label','韵律','砂厚/m','夹层频率','级差','夹层厚度']
df_jiaceng=df_jiaceng.dropna()
print(np.shape(df_jiaceng))
print('Class labels',np.unique(df_jiaceng['Class label']))
from sklearn.base import clone
from itertools import combinations
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
X,y=df_jiaceng.iloc[:,1:].values,df_jiaceng.iloc[:,0].values
print(np.bincount(y)[1:])
#划分测试数据和训练数据                                           
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)
#标准化
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std=stdsc.fit_transform(X_train)
X_test_std=stdsc.transform(X_test)
X_std = stdsc.transform(X)
X =X_std 
#计算Fscore
score = []
feat_labels = df_jiaceng.columns[1:]
mean_B=0
mean_S = 0
for j in range(X.shape[1]):
    for label in np.unique(y):
        """3类别
        j为样本第j个特征"""
        a= np.mean((X[y == label,j]))
        b = np.mean(X[:,j])
        c=np.square(a-b)
        mean_S=mean_S+c
    for label in np.unique(y):
        for k in  X[y == label,j]:
            aa=(np.square(k-np.mean(X[y == label,j],axis=0)))/(X[y == label].shape[0]-1)
            mean_B=mean_B+aa
    score.append(mean_S/mean_B)
indices = np.argsort(score)[::-1]
for f in range(X_train_std.shape[1]):
    print("%2d) %-*s %f" %(f+1,30,feat_labels[indices[f]],score[indices[f]]))
plt.title('F-score与特征权重')
plt.plot(range(X_train.shape[1]),score,marker = 'o')
plt.ylim([0,0.5])
plt.xticks(range(X_train.shape[1]),feat_labels,rotation = 90)
plt.xlim(-1,X_train.shape[1])
plt.tight_layout()
plt.grid()
plt.show()









