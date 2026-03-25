# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:24:57 2026

@author: PC
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve,auc
#分类问题
def Classification(y_pre, th):
    y_pred_class = []#命名规范
    for i in y_pre:
        if i >= th:#if语句别写错了
            y_pred_class.append(1)
        else:
            y_pred_class.append(0)
    return y_pred_class

#Confusion Matrix
def calculate_Confusion_Matrix(y_true, y_pre, th):
    y_pred_class = Classification(y_pre, th)
    Matrix=[[0,0],[0,0]]
    for i in range(len(y_pre)):
        Matrix[y_true[i]][y_pred_class[i]]+=1
    return Matrix

#Recall
def calculate_Recall(y_true,y_pre,th,Matrix):
    if np.sum(Matrix)==0:
        Matrix=calculate_Confusion_Matrix(y_true, y_pre, th)
    return Matrix[1][1]/(Matrix[1][0]+Matrix[1][1])


#FPR
def calculate_FPR(y_true,y_pre,th,Matrix):
    if np.sum(Matrix)==0:
        Matrix=calculate_Confusion_Matrix(y_true, y_pre, th)
    return Matrix[0][1]/(Matrix[0][1]+Matrix[0][0])

#排序
def sortyy(b,a):
    for i in range(len(b)):
        for j in range(1,len(b)-i):
            if a[j-1]<a[j]:
                a[j-1],a[j]=a[j],a[j-1]
                b[j-1],b[j]=b[j],b[j-1]
    return b,a
#得到数据
def get_ROC_data(y_true,y_score):
    y_true,y_score=sortyy(y_true,y_score)
    TPR=[]
    FPR=[]
    th=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    for i in th:
        Matrix=calculate_Confusion_Matrix(y_true, y_score,i)
        TPR.append(calculate_Recall(y_true, y_score,i,Matrix))
        FPR.append(calculate_FPR(y_true, y_score,i,Matrix))
    return TPR,FPR


#画图
def draw(X,Y,label):
    plt.title('ROC curve',fontsize=16)
    plt.plot(X,Y,'ro',label=label)
    plt.plot(X,Y)
    plt.ylabel('TPR',fontsize=16)
    plt.xlabel('FPR',fontsize=16)

#画ROC曲线
def draw_ROC(y_true,y_score):
    #print(y_score)
    TPR,FPR=get_ROC_data(y_true,y_score)
    #print(TPR)
    print(calculate_auc(FPR, TPR))
    draw(FPR,TPR,"ROC curve")

#计算auc
def calculate_auc(X,Y):
    print(X)
    print(Y)
    ans=0
    for i in range(1,len(X)):
        ans+=0.5*(X[i]-X[i-1])*(Y[i]+Y[i-1])
    return ans

#用机器学习进行画图
def draw_ROC_by_sklearn_learn(y_true,y_score):
    fpr,tpr,th=roc_curve(y_true,y_score)
    plt.plot(fpr, tpr, 'g--', lw=2)
    print("sklearn")
    print(auc(fpr,tpr))



if __name__ =="__main__":
    y_true=[1,1,0,1,0,0,1,0,0,0]
    y_score=[0.90,0.42,0.2,0.6,0.5,0.41,0.7,0.4,0.65,0.35]
    #y_true=[1,1,1,1,1,1,1,1,1,0]
    #y_score=[0.9,0.8,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    plt.figure(figsize=(5,5))
    draw_ROC(y_true,y_score)
    draw_ROC_by_sklearn_learn(y_true,y_score)


'''
1.自己写的 ROC 曲线
使用固定阈值（0.1 ~ 1）
点比较少（只有 10 个）
曲线不平滑
AUC 是你自己用梯形法算的
2. sklearn 的 ROC 曲线
使用所有可能的阈值（由数据自动生成）
点很多（通常 = 样本数）
曲线平滑、标准
AUC 是官方函数计算的

'''



