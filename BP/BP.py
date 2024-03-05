# -*- coding=utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import os
import sys

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
#随机划分训练验证集 # Randomly divide the training verification set
def splitTrainTestData(DataX,DataY,Data_inFID,trainDataRatio,TestDataRatio): 
    train_samples,val_samples,test_samples = [],[],[]
    labelsDic,classes={},[]#kb labelsDic是每个类别的个数
    for i in range(len(DataY)):
        label=DataY[i]
        if labelsDic.get(label)==None: 
                labelsDic[label]=1
        else:
                labelsDic[label]+=1
    
    dataSeparation=[trainDataRatio,1-trainDataRatio-TestDataRatio,TestDataRatio]
    for k in labelsDic:
            oneClass = [k, #kb对每个类别的样本 分成3份，
                    round(labelsDic[k] * dataSeparation[0]), 
                    round(labelsDic[k] * dataSeparation[1]),
                    int(labelsDic[k]) - round(labelsDic[k] * dataSeparation[0]) - round(labelsDic[k] * dataSeparation[1])]
            classes.append(oneClass)
    classes=np.array(classes)    #classes每一行存储 一类样本在训练、（验证）、测试数据中的数量 
    train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID=[],[],[],[],[],[],[],[],[]
    for i in range(len(DataY)):
        label=DataY[i]#getLabelOFIntFromLst(oneLabelLst)
        index=np.argwhere(classes[:,0]==label)[0][0].astype(np.int64)      
        if (classes[index][1] > 0):#kb[index][1]表示classes第index行对应类别的样本，在训练数据的个数
                train_DataX.append(DataX[i])
                train_DataY.append(DataY[i])
                train_Data_inFID.append(Data_inFID[i])
                # train_samples.append(oneSample.tolist())
                classes[index][1] = classes[index][1]-1
        elif (classes[index][2] > 0):#kb[index][2]表示classes第index行对应类别的样本，在验证数据的个数
                val_DataX.append(DataX[i])
                val_DataY.append(DataY[i])
                val_Data_inFID.append(Data_inFID[i])
                # val_samples.append(oneSample.tolist())
                classes[index][2] = classes[index][2]-1
        else:
                test_DataX.append(DataX[i])
                test_DataY.append(DataY[i])
                test_Data_inFID.append(Data_inFID[i])
                # test_samples.append(oneSample.tolist())
                classes[index][3] = classes[index][3]-1
    train_DataX,train_DataY,train_Data_inFID=np.array(train_DataX),np.array(train_DataY),np.array(train_Data_inFID)
    val_DataX,val_DataY,val_Data_inFID=np.array(val_DataX),np.array(val_DataY),np.array(val_Data_inFID)
    test_DataX,test_DataY,test_Data_inFID=np.array(test_DataX),np.array(test_DataY),np.array(test_Data_inFID)

    return train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID
# 划分训练、验证、测试集:训练集中各类样本数量相等、验证集中各类样本数量相等，剩余数据为测试集。
# 如：训练中各类样本数量均为80，验证集均为20。
# Split the dataset into training, validation, and testing sets: Equal number of samples for each class in the training and validation sets, remaining data for testing.
# For example: 80 samples for each class in the training set, 20 for validation.
def splitTrainValTestByCountRandom(x, y, id, lst_Num_singleCls_train, lst_Num_singleCls_val, index_arr):
    # train_DataX, train_DataY, train_Data_inFID, val_DataX, val_DataY, val_Data_inFID, test_DataX, test_DataY, test_Data_inFID = (), (), (), (), (), (), (), (), ()
    train_DataX, train_DataY, train_Data_inFID, val_DataX, val_DataY, val_Data_inFID, test_DataX, test_DataY, test_Data_inFID = [], [], [], [], [], [], [], [], []
    SampleNum = len(y)
    if index_arr is None:
        index_arr = [i for i in range(SampleNum)]  # 得到一个元素为0，1，2...SampleNum-1的数组（Array with indices 0 to SampleNum-1）
        import random
        random.shuffle(index_arr)  # 随机打乱数组index_arr内的元素，达到随机选择样本的目的（Shuffle the indices for random sample selection）
    numclass = len(np.unique(y))
    y_one_hot = tf.one_hot(y, depth=numclass)  # 标签转换为onehot（Convert labels to one-hot encoding）
    with tf.Session() as sess:
        y_one_hot = y_one_hot.eval()
        labels_Num_Train_dic = {}
        labels_Num_Val_dic = {}
        for i in range(numclass):
            labels_Num_Train_dic[i] = lst_Num_singleCls_train[i]  # labels_Num_Train_dic[i]表示训练集中类别为i的样本的数量（labels_Num_Train_dic[i] indicates the number of samples for class i in the training set）
            labels_Num_Val_dic[i] = lst_Num_singleCls_val[i]
        for i in range(len(index_arr)):
            index_i = index_arr[i]  # index_i是0-SampleNum的索引被随机打乱的索引（index_i is the randomly shuffled index from 0 to SampleNum-1）
            y_i = y[index_i]
            if labels_Num_Train_dic[y_i] > 0:
                train_DataX.append(x[index_i])
                train_DataY.append(y_one_hot[index_i])
                train_Data_inFID.append(id[index_i])
                labels_Num_Train_dic[y_i] = labels_Num_Train_dic[y_i] - 1
            elif labels_Num_Val_dic[y_i] > 0:
                val_DataX.append(x[index_i])
                val_DataY.append(y_one_hot[index_i])
                val_Data_inFID.append(id[index_i])
                labels_Num_Val_dic[y_i] = labels_Num_Val_dic[y_i] - 1
            else:
                test_DataX.append(x[index_i])
                test_DataY.append(y_one_hot[index_i])
                test_Data_inFID.append(id[index_i])
    return train_DataX, train_DataY, train_Data_inFID, val_DataX, val_DataY, val_Data_inFID, test_DataX, test_DataY, test_Data_inFID

# Output the results of each sample's prediction
def OutputPredictRes(inFID_Lst, real_Label_Lst, predict_Proba, predict_Label, output_Path):
    output_Lst = []
    for i in range(len(inFID_Lst)):
        output_Lst_i = []
        output_Lst_i.append(inFID_Lst[i])
        output_Lst_i.append(real_Label_Lst[i])
        output_Lst_i.append(predict_Label[i])
        output_Lst_i += predict_Proba[i]
        output_Lst.append(output_Lst_i)
    np.savetxt(output_Path, output_Lst, fmt='%s')

# 绘制混淆矩阵Plot the confusion matrix showing percentages or values
def plot_confusion_matrix_percent(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                                  confusionMatrixPngPath='./confusionMatrixPng.png'):
    """
    - cm: 计算出的混淆矩阵的值（Computed confusion matrix values）
    - classes: 混淆矩阵中每一行每一列对应的列（Columns corresponding to each row and column in the confusion matrix）
    - normalize: True:显示百分比, False:显示个数（True: Show percentages, False: Show counts）
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

def BPAlgorithm(train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID,n_Hidden,n_Output,training_Epochs,learning_rate,save_Path,outFileName_BasedParam): #训练数据 测试数据 隐藏层神经元个数 输出层神经元个数 迭代次数 学习率 结果保存路径
    traindata_Columns = np.size(train_DataX,1)
    n_Input = traindata_Columns #输入层神经元个数
    input_X = tf.placeholder(dtype=tf.float32,shape=[None,n_Input])
    input_Y = tf.placeholder(dtype=tf.float32,shape=[None,n_Output])

    W1 = tf.Variable(tf.random.normal([n_Input,n_Hidden],seed=0))
    b1 = tf.Variable(tf.random.normal(shape=[n_Hidden],seed=0))
    hidden_Layer = tf.nn.sigmoid(tf.add(tf.matmul(input_X,W1),b1))
    
    W2 = tf.Variable(tf.random.normal([n_Hidden,n_Output],seed=0))
    b2 = tf.Variable(tf.random.normal(shape=[n_Output],seed=0))
    predict = tf.nn.softmax(tf.add(tf.matmul(hidden_Layer,W2),b2))
    
    loss = tf.reduce_mean(tf.square(input_Y-predict))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(predict,1),tf.argmax(input_Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    train_acc,train_cost,train_Predict,val_acc ,val_cost,val_Predict=0,0,0,0,0,0
    x_epoch, accs_train,accs_val=[],[],[]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        preLoss = 0
        for i in range(1,training_Epochs + 1):
            _,sumLoss = sess.run([train,loss],feed_dict={input_X:train_DataX,input_Y:train_DataY})
            if i == 1 or i % 1000 == 0: #每2000次打印
                # print('hidden:%i  step:%i  loss: %f'%(n_Hidden,i,sumLoss))
                train_acc,train_cost,train_Predict = sess.run([accuracy,loss,predict], feed_dict={input_X:train_DataX,input_Y:train_DataY})
                val_acc ,val_cost,val_Predict= sess.run([accuracy,loss,predict],feed_dict={input_X:val_DataX,input_Y:val_DataY})
                x_epoch.append(i)
                accs_train.append(train_acc)
                accs_val.append(val_acc)
                print('Step %i, Train_Loss: %f, Train_Acc: %s, Train_Loss: %f, val_Acc: %s' % (i, sumLoss, train_acc,val_cost, val_acc))
                # if i != 1 and abs(sumLoss - preLoss) < 0.0000001: #两次打印得到的loss值过小 就break
                #     break       
            preLoss = sumLoss
        #得到 数据集 的各评价指标
        train_DataY2=tf.argmax(train_DataY,1).eval()#[np.argmax(i) for i in train_DataY]
        train_Predict_Y=tf.argmax(train_Predict,1).eval()#np.argmax(train_Predict, axis=1)
        precision_train=precision_score(train_DataY2,train_Predict_Y,average='weighted')#macro不考虑类别不均衡的影响；
        f1_train = f1_score(train_DataY2,train_Predict_Y,average='weighted')
        roc_auc_train=roc_auc_score(train_DataY2,train_Predict,average='weighted',multi_class='ovr')
        val_DataY2=tf.argmax(val_DataY,1).eval()
        val_Predict_Y=np.argmax(val_Predict,axis=1)
        precision_val=precision_score(val_DataY2,val_Predict_Y,average='weighted')
        f1_val = f1_score(val_DataY2,val_Predict_Y,average='weighted')
        roc_auc_val=roc_auc_score(val_DataY2,val_Predict,average='weighted',multi_class='ovr')
        # train_Predict,train_acc = sess.run([predict,accuracy],feed_dict={input_X:train_DataX,input_Y:train_DataY})
        # val_Predict,val_acc = sess.run([predict,accuracy],feed_dict={input_X:val_DataX,input_Y:val_DataY})
        test_Predict,test_acc = sess.run([predict,accuracy],feed_dict={input_X:test_DataX,input_Y:test_DataY})
        test_DataY2=tf.argmax(test_DataY,1).eval()
        test_Predict_Y=tf.argmax(test_Predict,1).eval()#np.argmax(test_Predict,axis=1)
        precision_test=precision_score(test_DataY2,test_Predict_Y,average='weighted')
        f1_test = f1_score(test_DataY2,test_Predict_Y,average='weighted')
        roc_auc_test=roc_auc_score(test_DataY2,test_Predict,average='weighted',multi_class='ovr')
        out_acc_lst=['accuracy',train_acc ,val_acc,test_acc,'precision',precision_train,precision_val,precision_test,'f1score',f1_train,f1_val,f1_test,'rocauc',roc_auc_train,roc_auc_val,roc_auc_test]
        attack_types = ['Industrial','Commercial', 'Resident', 'Public', 'Education', 'Mixed', 'UrbanVillage']
        #训练样本的 混淆矩阵
        cm_train= confusion_matrix(train_DataY2,train_Predict_Y)
        plot_confusion_matrix_percent(cm_train, classes=attack_types, normalize=False, title='Normalized confusion matrix  train_acc '+str(round(train_acc.item(),4)),confusionMatrixPngPath=save_Path+"\\CMtrain"+outFileName_BasedParam+".png")
        #测试样本的 混淆矩阵
        cm_test= confusion_matrix(test_DataY2,test_Predict_Y)
        plot_confusion_matrix_percent(cm_test, classes=attack_types, normalize=False, title='Normalized confusion matrix  test_acc '+str(round(test_acc.item(),4)),confusionMatrixPngPath=save_Path+"\\CMtest"+outFileName_BasedParam+".png")
        #将预测结果输出到txt
        OutputPredictRes(train_Data_inFID,train_DataY2,train_Predict.tolist(),tf.argmax(train_Predict,1).eval(),save_Path+'\\train_result'+outFileName_BasedParam+'.txt')
        OutputPredictRes(val_Data_inFID,val_DataY2,val_Predict.tolist(),tf.argmax(val_Predict,1).eval(),save_Path+'\\val_result'+outFileName_BasedParam+'.txt')
        OutputPredictRes(test_Data_inFID,test_DataY2,test_Predict.tolist(),tf.argmax(test_Predict,1).eval(),save_Path+'\\test_result'+outFileName_BasedParam+'.txt')
        np.savetxt(save_Path+'\\acc'+outFileName_BasedParam+'.txt',out_acc_lst,fmt="%s",delimiter=',')#将分类概率保存到txt
        #得到 测试数据 的分类准确度
        print('测试准确度')
        print(test_acc)

        #绘制 曲线  
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')#, color=color)
        ax1.plot(x_epoch, accs_train,color='red')#exp_moving_avg(accs_train),color='red')
        ax1.plot(x_epoch, accs_val, color='blue')#exp_moving_avg(accs_val), color='blue')
        ax1.tick_params(axis='y')#, labelcolor=color)
        ax1.set_title(outFileName_BasedParam)
        plt.legend(['Training accuracy ', 'Validation accuracy'])

        plt.savefig(save_Path+"/"+'curves '+outFileName_BasedParam+".png")
        # plt.show()
        plt.close()
    return test_acc


def main():
    #样本的特征、标签、ID路径# Sample features, tags, ID paths
    fea_np= np.loadtxt(r'data/feature.txt',dtype= np.float32,delimiter=' ',encoding='utf-8')
    label= np.loadtxt(r'data/y.txt',dtype= np.int32,delimiter=' ',encoding='utf-8')
    TagID=np.loadtxt(r'data/TagID.txt',dtype= np.str_,encoding='utf-8')
    save_Path=r'data' 
    lst_Num_singleCls_train,lst_Num_singleCls_val=[100,300,300,300,300,300,300],[50,100,100,100,100,100,100]#[50,100,100,100,100,100,100]
    train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID=None,None,None,None,None,None,None,None,None
    #随机划分训练、验证、测试集# Randomly divide training, verification, and test sets
    outFileFolder=r'data'
    index_arr=np.load(outFileFolder+'/'+'random_index_arr.npy')#提前随机生成的数据的ID，避免多次随机生成的训练样本不一样。# Advance the ID of randomly generated data to avoid multiple randomly generated training samples are not the same.
    train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID=splitTrainValTestByCountRandom(fea_np,label,TagID,lst_Num_singleCls_train,lst_Num_singleCls_val,index_arr)
    train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID=np.array(train_DataX),np.array(train_DataY),np.array(train_Data_inFID),np.array(val_DataX),np.array(val_DataY),np.array(val_Data_inFID),np.array(test_DataX),np.array(test_DataY),np.array(test_Data_inFID)
    #参数
    Hidden_lst=[20]  
    learning_rate_lst=[0.03]
    training_Epochs=1000
    n_Output=len(train_DataY[0]) #输出维度：分类数量
    for Hidden_i in range(len(Hidden_lst)):
        for learning_rate_j in range(len(learning_rate_lst)):
            n_Hidden=Hidden_lst[Hidden_i]
            learning_rate=learning_rate_lst[learning_rate_j]
            outFileName_BasedParam='_n_Hidden'+str(n_Hidden)+'_learning_rate'+str(learning_rate)
            #开始训练、预测# Start training and forecasting
            BPAlgorithm(train_DataX,train_DataY,train_Data_inFID,val_DataX,val_DataY,val_Data_inFID,test_DataX,test_DataY,test_Data_inFID,n_Hidden,n_Output,training_Epochs,learning_rate,save_Path,outFileName_BasedParam)  

if __name__ == "__main__":
    main()