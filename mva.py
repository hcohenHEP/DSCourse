"""
======================================================
Multi Variate Analysis(MVA)
======================================================
This model is responsible for the ML algorithm training and predictions.
Using sklearn this model will train the BDT and once new data arrives,
predict it's classification
"""
import os
import sys
import pandas as pd
import matplotlib
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#The features used for the training & prediction of the versions
features = ['AvgSessionDuration','DTW_FV','DTW_FS','DTW_SV',
            'FailAvgDuration','TotalAvgDuration']

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    """Confusion Matrix Plotting - taken from the course's demo
    :param cm: Confusion matrix
    :param labels: the classification labels
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def train():
    """Main BDT training method. Loads the completely mined data from the file
       Select proper portion of the data for training, trains a BDT, saves
       the model for future use and visualize the BDT

       Function does not return anything but shows the confusion matrices
       of the training & testing for the BDT
    """
    df = pd.read_csv('DataLabeled.csv' ,encoding='latin_1', index_col=0,
                     parse_dates=False, low_memory=False)
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], -1)
    y_truth = np.array(df["Class"])
    sig_len = len(y_truth[y_truth==1])
    bkg_len = len(y_truth[y_truth==0])
    global features
    df = df[features]
    dataset = np.array(df)
    labels=["Good","Faulty"]

    #Since the amount of data is very very limited we need to have a
    #significant portion to test the algorithm on
    pct_test = 0.45
    #Randomize the sample selection and make sure to have enough in each class
    num_sig_train = math.floor(pct_test * sig_len)
    num_sig_test = len(y_truth[y_truth==1]) - num_sig_train
    num_bkg_test = math.floor(pct_test * bkg_len)
    num_bkg_train = bkg_len-num_bkg_test

    train_sig_i = np.random.choice(sig_len,num_sig_train,replace=False)
    test_sig_i = np.setdiff1d(np.arange(sig_len),train_sig_i)
    train_bkg_i = np.random.choice(bkg_len,num_bkg_train,replace=False) + sig_len
    test_bkg_i = np.setdiff1d(np.arange(bkg_len)+sig_len,train_bkg_i)
    train_i = np.append(train_sig_i,train_bkg_i)
    test_i = np.append(test_sig_i,test_bkg_i)
    train_set = dataset[train_i]
    train_labels = y_truth[train_i]
    test_set = dataset[test_i]
    test_labels = y_truth[test_i]

    #dt_cls = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_weight_fraction_leaf=0.01)
    dt_cls = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='entropy',
                                                            max_depth=3, min_weight_fraction_leaf=0.01),
                         algorithm="SAMME",
                         n_estimators=100)
    dt_cls.fit(train_set, train_labels)
    pickle.dump(dt_cls,open('trained_bdt.pkl', 'wb'))

    #Export 1 out off 100 trees just for visualization
    with open("bdt.dot", 'w') as f:
        f = tree.export_graphviz(dt_cls.estimators_[0], out_file=f, feature_names=list(df.columns), class_names=labels, filled=True, impurity=True, rounded=True)

    #Make predictions and test accuracy of the model
    train_cls = dt_cls.predict(train_set)
    test_cls = dt_cls.predict(test_set)
    tr_acc = accuracy_score(train_labels, train_cls) * 100
    ts_acc = accuracy_score(test_labels, test_cls) * 100

    #Training CM
    confusion_mat = confusion_matrix(train_labels, train_cls)
    confusion_mat = confusion_mat.astype('float')*100.0/confusion_mat.sum(axis=1)
    print('Training Set Confusion matrix')
    print(confusion_mat)
    plt.figure()
    plot_confusion_matrix(confusion_mat,labels, title='Training Set Confusion matrix,')

    #Testing CM
    confusion_mat = confusion_matrix(test_labels, test_cls)
    confusion_mat = confusion_mat.astype('float')*100.0/confusion_mat.sum(axis=1)
    print('Testing Set Confusion matrix')
    print(confusion_mat)
    plt.figure()
    plot_confusion_matrix(confusion_mat,labels, title='Testing Set Confusion matrix,')

    print('Accuracy: Training=', tr_acc, ', Testing=', ts_acc)

    plt.show()


def predict(df):
    """Provides prediction from previously trained model to new data
       Returns the origin df with extra column of 'Class' with the proper prediction
       1=Faulty Version, 0=Good Version


    :param df: data frame with mined data
    """
    global features
    dataset = np.array(df[features])
    dt_cls = pickle.load(open('trained_bdt.pkl', 'rb'))
    prediction = dt_cls.predict(dataset)
    df["Class"] = prediction
    return df
