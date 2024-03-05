# -*- coding=utf-8 -*-
import math
from operator import contains
from numpy.core.defchararray import title
from scipy.sparse.construct import random
import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from random import random
import time
import os
import sys
from matplotlib import pylab as plt

from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np

from sklearn.datasets import load_iris
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.metrics import confusion_matrix
#根据样本被归类为各类别的概率yPredict_proba，得到样本的归类类别；再根据样本的真实类别yTrue，得到分类准确度
# (# According to the probability of the sample being classified into various categories yPredict_proba, the classification category of the sample is obtained; Then, the classification accuracy is obtained according to the true category yTrue of the sample)
def getAccuracyFromProba(yTrue,yPredict_proba):
    yPredict=[]#根据各类别的分类概率，得到的分类类别
    for i in range(len(yPredict_proba)):
        maxILabel=9999
        maxProba=-9999
        for j in range(len((yPredict_proba[i]))):
            if maxProba<yPredict_proba[i][j]:
                maxProba=yPredict_proba[i][j]
                maxILabel=j
        if maxILabel!=9999:
            yPredict.append(maxILabel)
    predTrueNum=0#分类正确的个数
    for i in range(len(yTrue)):
        if yTrue[i]==yPredict[i]:
            predTrueNum=predTrueNum+1
    acc=predTrueNum/len(yTrue)
    return acc,yPredict

#将每个样本的预测结果输出(# Output the prediction results for each sample)
def OutputPredictRes(inFID_Lst,real_Label_Lst,predict_Proba,predict_Label,output_Path):
    output_Lst=[]
    for i in range(len(inFID_Lst)):
        output_Lst_i=[]
        output_Lst_i.append(inFID_Lst[i])
        output_Lst_i.append(real_Label_Lst[i])
        output_Lst_i.append( predict_Label[i])
        output_Lst_i+=predict_Proba[i]
        output_Lst.append(output_Lst_i)
    np.savetxt(output_Path,output_Lst,fmt='%s')

#划分训练、验证、测试集:训练集中各类样本数量相等、验证集中各类样本数量相等，剩余数据为 测试集。
# 如：训练中各类样本数量均为80，验证集均为20.  
#(# Divide the training, verification and test sets: the number of samples in the training set is equal, the number of samples in the verification set is equal, and the remaining data is the test set.
# For example: the number of all types of samples in training is 80, and the verification set is 20.)  
def splitTrainValTestByCountRandom(x,y,id,lst_Num_singleCls_train,lst_Num_singleCls_val,index_arr,trainTagID,valTagID,testTagID):
    # train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID=(),(),(),(),(),(),(),(),()
    train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID=[],[],[],[],[],[],[],[],[]
    SampleNum=len(y)    
    numclass=len(np.unique(y))
    labels_Num_Train_dic={}
    labels_Num_Val_dic={}
    for i in range(numclass):
        labels_Num_Train_dic[i]=lst_Num_singleCls_train[i] #labels_Num_Train_dic[i]表示训练集中 类别为i的样本的数量
        labels_Num_Val_dic[i]=lst_Num_singleCls_val[i] 
    for i in range(len(id)): 
        if id[i] in trainTagID:
            train_DataX.append(x[i])
            train_DataY.append(y[i])
            train_Data_inFID.append(id[i])
        elif id[i] in valTagID:
            val_DataX.append(x[i])
            val_DataY.append(y[i])
            val_Data_inFID.append(id[i])
        else:
            test_DataX.append(x[i])
            test_DataY.append(y[i])
            test_Data_inFID.append(id[i])
    return train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID

# 绘制混淆矩阵 显示百分比或数值(# Draw the confusion matrix to show percentages or values)
def plot_confusion_matrix_percent(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,confusionMatrixPngPath='./confusionMatrixPng.png'):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("混淆矩阵：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('混淆矩阵：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig(confusionMatrixPngPath)
    plt.close()

#Random Forest train and predict
def TrainAndPredict(xtrain, ytrain,train_Data_inFID,xval=None,yval=None,val_Data_inFID=None,xtest=None,ytest=None,test_Data_inFID=None,TestDistrict1X=None,TestDistrict1Y=None,TestDistrict1_inFID=None,TestDistrict2X=None,TestDistrict2Y=None,TestDistrict2_inFID=None,outFileFolder='',outFileName_BasedParam=''):
    annoLst=[
        'RandomForest']
    n_estimators,max_features,max_depth,max_leaf_nodes =100,4,16,128
    #clf1=RandomForestClassifier(max_depth=8,max_leaf_nodes=16,random_state=1)#n_estimators=100（弱分类器个数，没啥影响）默认100；min_samples_leaf影响不大；oob_score是否采用袋外样本来评估模型的好坏(泛化能力)默认False，改为True无变化
    ##n_estimators=100（弱分类器个数，没啥影响）默认100；min_samples_leaf影响不大；oob_score是否采用袋外样本来评估模型的好坏(泛化能力)默认False，改为True无变化
    clf1=RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,random_state=1)
    outFileName_BasedParam='_n_estimators'+str(n_estimators)+'_max_features'+str(max_features)+'_max_depth'+str(max_depth)+'_max_leaf_nodes'+str(max_leaf_nodes)
    out_acc_lst=[]
    #混淆矩阵的准备工作(# Preparation of the confusion matrix)
    attack_types = ['Industrial','Commercial', 'Resident', 'Public', 'Education', 'Mixed', 'UrbanVillage']
    for clf, label in zip([clf1], 
                    ['',#'Random Forest'
                     ]):#'StackingClassifier']):
        clf.fit(xtrain, ytrain) 
        
        p_sclf=clf.predict_proba(xtrain)#获得训练数据的分类概率 
        acc,yPredict= getAccuracyFromProba(ytrain,p_sclf) 
        out_acc_lst.append(acc)

        output_Path=outFileFolder+'\\train'+label+outFileName_BasedParam+'.txt'
        OutputPredictRes(train_Data_inFID.tolist(),ytrain.tolist(),np.mat(p_sclf).tolist(),yPredict,output_Path)#将每个block的预测结果、概率 输出
        precision_train=precision_score(ytrain.tolist(),yPredict,average='weighted')#macro不考虑类别不均衡的影响；
        f1_train = f1_score(ytrain.tolist(),yPredict,average='weighted')
        # roc_auc_train=roc_auc_score(ytrain.tolist(),p_sclf,average='weighted',multi_class='ovr')
        #混淆矩阵   训练数据
        cm_train= confusion_matrix(ytrain,yPredict)
        plot_confusion_matrix_percent(cm_train, classes=attack_types, normalize=False, title='Normalized confusion matrix  train_acc '+str(round(acc,4)),confusionMatrixPngPath=outFileFolder+"\\CMtrain"+outFileName_BasedParam+".png")
        print()
        p_sclf_val,acc_val='',''
        if xval is not None:
            p_sclf_val=clf.predict_proba(xval)#获得验证数据的分类概率 
            acc_val,yPredict_val= getAccuracyFromProba(yval,p_sclf_val) 
            out_acc_lst.append(acc_val)
            output_Path=outFileFolder+'\\val'+label+outFileName_BasedParam+'.txt'
            OutputPredictRes(val_Data_inFID.tolist(),yval.tolist(),np.mat(p_sclf_val).tolist(),yPredict_val,output_Path)
            precision_val=precision_score(yval.tolist(),yPredict_val,average='weighted')
            f1_val = f1_score(yval.tolist(),yPredict_val,average='weighted')
            # roc_auc_val=roc_auc_score(yval.tolist(),p_sclf_val,average='weighted',multi_class='ovr')
        print()
        p_sclf_test,acc_test='',''
        if xtest is not None:
            p_sclf_test=clf.predict_proba(xtest)#获得测试数据的分类概率 
            acc_test,yPredict_test= getAccuracyFromProba(ytest,p_sclf_test) 
            out_acc_lst.append(acc_test)
            output_Path=outFileFolder+'\\test'+label+outFileName_BasedParam+'.txt'
            OutputPredictRes(test_Data_inFID.tolist(),ytest.tolist(),np.mat(p_sclf_test).tolist(),yPredict_test,output_Path)
            precision_test=precision_score(ytest.tolist(),yPredict_test,average='weighted')
            f1_test = f1_score(ytest.tolist(),yPredict_test,average='weighted')
            # roc_auc_test=roc_auc_score(ytest.tolist(),p_sclf_test,average='weighted',multi_class='ovr')
            #混淆矩阵  测试数据
            cm_test= confusion_matrix(ytest,yPredict_test)
            plot_confusion_matrix_percent(cm_test, classes=attack_types, normalize=False, title='Normalized confusion matrix  test_acc '+str(round(acc_test,4)),confusionMatrixPngPath=outFileFolder+"\\CMtest"+outFileName_BasedParam+".png")
            
        #测试另外2个区域
        p_sclf_TestDistrict1,acc_TestDistrict1,p_sclf_TestDistrict2,acc_TestDistrict2='','','',''
        if TestDistrict1X is not None:
            p_sclf_TestDistrict1=clf.predict_proba(TestDistrict1X)#获得分类概率 TestDistrict1X,TestDistrict1Y,TestDistrict2X,TestDistrict2Y
            acc_TestDistrict1,yPredict_TestDistrict1= getAccuracyFromProba(TestDistrict1Y,p_sclf_TestDistrict1) 
            output_Path=outFileFolder+'\\TestDistrict1'+label+outFileName_BasedParam+'.txt'
            OutputPredictRes(TestDistrict1_inFID.tolist(),TestDistrict1Y.tolist(),np.mat(p_sclf_TestDistrict1).tolist(),yPredict_TestDistrict1,output_Path)
            f1_micro_TestDistrict1 = f1_score(TestDistrict1Y.tolist(),yPredict_TestDistrict1,average='weighted')
        if TestDistrict2X is not None:
            p_sclf_TestDistrict2=clf.predict_proba(TestDistrict2X)#获得分类概率 TestDistrict1X,TestDistrict1Y,TestDistrict2X,TestDistrict2Y                
            acc_TestDistrict2,yPredict_TestDistrict2= getAccuracyFromProba(TestDistrict2Y,p_sclf_TestDistrict2) 
            output_Path=outFileFolder+'\\TestDistrict2'+label+outFileName_BasedParam+'.txt'
            OutputPredictRes(TestDistrict2_inFID.tolist(),TestDistrict2Y.tolist(),np.mat(p_sclf_TestDistrict2).tolist(),yPredict_TestDistrict2,output_Path)
            f1_micro_TestDistrict2 = f1_score(TestDistrict2Y.tolist(),yPredict_TestDistrict2,average='weighted')
    output_Path=outFileFolder+'\\acc'+label+outFileName_BasedParam+'.txt'
    out_acc_lst=['accuracy',acc ,acc_val,acc_test,'precision',precision_train,precision_val,precision_test,'f1score',f1_train,f1_val,f1_test]#,'rocauc',roc_auc_train,roc_auc_val,roc_auc_test]
    np.savetxt(output_Path,out_acc_lst,fmt='%s')
    print(annoLst[0])
    #打印precision
    print('preci_train: %0.4f '%precision_train,end="")
    if xval is not None:
        print('    preci_val  '+':%0.4f'%precision_val, end="")
    else:
        print('    *****', end="")
    if xtest is not None:
        print('    preci_test  '+':%0.4f'%precision_test, end="")
    else:
        print('    *****', end="")
    if TestDistrict1X is not None:
        print('   光明区%0.4f'%acc_TestDistrict1, end="")
    else:
        print('    *****', end="")
    if TestDistrict2X is not None:
        print('   福田区%0.4f'%acc_TestDistrict2)
    else:
        print('    *****')
    #打印acc准确度
    print('acc_train: %0.4f '%acc,end="")
    if xval is not None:
        print('    acc_val  '+':%0.4f'%acc_val, end="")
    else:
        print('    *****', end="")
    if xtest is not None:
        print('    acc_test  '+':%0.4f'%acc_test, end="")
    else:
        print('    *****', end="")
    if TestDistrict1X is not None:
        print('   光明区%0.4f'%acc_TestDistrict1, end="")
    else:
        print('    *****', end="")
    if TestDistrict2X is not None:
        print('   福田区%0.4f'%acc_TestDistrict2)
    else:
        print('    *****')
    #打印f1
    print('f1_train: %0.4f '%f1_train,end="")
    if xval is not None:
        print('    f1_val  '+':%0.4f'%f1_val, end="")
    else:
        print('    *****', end="")
    if xtest is not None:
        print('    f1_test  '+':%0.4f'%f1_test, end="")
    else:
        print('    *****', end="")
    if TestDistrict1X is not None and f1_micro_TestDistrict1 is not None:
        print('   光明区%0.4f'%f1_micro_TestDistrict1, end="")
    else:
        print('    *****', end="")
    if TestDistrict2X is not None and f1_micro_TestDistrict2 is not None:
        print('   福田区%0.4f'%f1_micro_TestDistrict2)
    else:
        print('    *****')

def main():
    #样本的特征、标签、ID路径 (# Sample features, tags, ID paths)
    fea_np= np.load(r'data\RF-SI-BuildingFeanpy_.npy')
    #样本集划分与SA1-4在本文方法中的划分保持一致(# Sample set partitioning is consistent with the partitioning of SA1-4 in this paper's method)
    trainTagID= np.load(r'data\TrainTagID.npy')  
    valTagID= np.load(r'data\ValTagID.npy')
    testTagID= np.load(r'data\TestTagID.npy')
    # fea_np= np.loadtxt(r'data\featureMatrix_检查标注SA1_2_10m_Del500M_有工业.txt',dtype= np.float32,delimiter=' ',encoding='utf-8')
    label= np.loadtxt(r'data\Ally.txt',dtype= np.int32,delimiter=' ',encoding='utf-8')
    TagID=np.loadtxt(r'data\AllTagID.txt',dtype= np.str_,encoding='utf-8')
    lst_Num_singleCls_train,lst_Num_singleCls_val=[100,300,300,300,300,300,300],[50,100,100,100,100,100,100]
    train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID=None,None,None,None,None,None,None,None,None
    #随机划分训练、验证、测试集,生成输入数据。(# Randomly divide training, verification, and test sets to generate input data.)
    outFileFolder=r'data'
    index_arr=np.load(outFileFolder+'/'+'random_index_arr.npy')
    train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID=splitTrainValTestByCountRandom(fea_np,label,TagID,lst_Num_singleCls_train,lst_Num_singleCls_val,index_arr,trainTagID,valTagID,testTagID)
    train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID=np.array(train_DataX),np.array(train_DataY),np.array(train_Data_inFID),np.array(val_DataX),np.array(val_DataY),np.array(val_Data_inFID),np.array(test_DataX),np.array(test_DataY),np.array(test_Data_inFID)
    outFileFolder=r'data' 
    outFileName_BasedParam=''
    # #另外2个用于测试的区的数据(# # Data for the other 2 areas used for testing, not used in this code.)
    TestDistrict1X,TestDistrict1Y,TestDistrict1_inFID,TestDistrict2X,TestDistrict2Y,TestDistrict2_inFID=None,None,None,None,None,None
    #训练和预测(# Training and prediction)
    TrainAndPredict(train_DataX,train_DataY,train_Data_inFID,
            val_DataX,val_DataY,val_Data_inFID,
            test_DataX,test_DataY,test_Data_inFID,
            TestDistrict1X=TestDistrict1X,TestDistrict1Y=TestDistrict1Y,TestDistrict1_inFID=TestDistrict1_inFID,
            TestDistrict2X=TestDistrict2X,TestDistrict2Y=TestDistrict2Y,TestDistrict2_inFID=TestDistrict2_inFID,
            outFileFolder=outFileFolder,outFileName_BasedParam=outFileName_BasedParam)
    print('Complete!')

if __name__ == "__main__":
    main()

