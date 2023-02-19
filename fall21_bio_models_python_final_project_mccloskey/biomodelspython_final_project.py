# BIOL 5514 Final Project

# deepsqueak to process raw USV data -> excel files
# three classes: rat, call log, cohort
# machine learning to find patterns distinguishing calls from stress-resistant and stress-sensitive animals

# import modules
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn import svm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras import optimizers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from sklearn.externals import joblib # to save model after training

class Rat():
    # store dataframes from DeepSqueak output,
    # id, suppression ratio, and clustering info
    def __init__(self,ratid=None,sr=None,cluster=None,dfs={}):
        self.ratid = ratid
        self.sr = sr
        self.cluster = cluster
        self.dfs = dfs # empty dictionary to store dfs by phase

class Call_Log():    
    # store subset of data and train_test_split
    # build, train and test ML models
    def __init__(self,name=None,s=None,testsize=None):
        self.name = name
        self.s = s # folder list slice
        self.testsize = testsize # proportion of data to allocate for testing
        self.dfs = [] # from excel files of usv data
        self.Xtrain = None # training data
        self.ytrain = None
        self.Xtest = None # test data
        self.ytest = None
        self.svm_acc = None # accuracy from svm model for individual calls
        self.dt_acc = None # accuracy from decision tree model for individual calls
        self.tf_acc = None # accuracy from tensor flow model for individual calls
        self.tor_cm = None # confusion matrix of torch mzodel
        self.tor_acc = None # accuracy of torch model
    
    def merge_split_data(self,rats):
        # concatenate all relevant data from rats and split into train and test portions
        all_dfs = []
        # loop over all folders determined by call log name
        for f in self.s:
            # loop over all rat objects in cohort
            for rat in rats:
                # add all dataframes for each phase from each rat object
                all_dfs.extend([d for d in rat.dfs[f]])
        # merge all data
        all_data = pd.concat(all_dfs)
        # shuffle data to avoid getting stuck in local minimum
        all_data = all_data.sample(frac=1) 
        # slice only numerical features into X
        X = all_data.loc[:,'Score':'Peak Freq (kHz)']
        # normalize data
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        # category labels to y
        y = all_data['clabel']
        # 80/20 split if not otherwise specified
        if self.testsize == None:
            self.testsize = 0.2
        # train/test split
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(Xs, y, test_size=self.testsize, random_state=69)
        # scaler = StandardScaler()
        # self.Xtrain = scaler.fit_transform(Xtrain)
        # self.Xtest = scaler.fit_transform(Xtest)
    
    def SupportVectorMachine(self):
        # build and test SVM model
        model = svm.SVC()
        model.fit(self.Xtrain,self.ytrain)
        predictions = model.predict(self.Xtest)
        self.svm_acc = accuracy_score(self.ytest, predictions)
        
    def DecisionTree(self):
        # build and test decision tree model
        model = DecisionTreeClassifier()
        model.fit(self.Xtrain, self.ytrain)
        predictions = model.predict(self.Xtest)
        self.dt_acc = accuracy_score(self.ytest, predictions)
        # export image
        tree.export_graphviz(model, out_file='file.dot', feature_names=['Score','Begin Time (s)','End Time (s)','Call Length (s)','Principal Frequency (kHz)','Low Freq (kHz)','High Freq (kHz)','Delta Freq (kHz)','Frequency Standard Deviation (kHz)','Slope (kHz/s)','Sinuosity','Mean Power (dB/Hz)','Tonality','Peak Freq (kHz)'], class_names=['punishment-resistant','punishment-sensitive'],label='all', rounded=True, filled=True)

    def TensorFlow(self):
        # source: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
        X_train = self.Xtrain
        y_train = self.ytrain
        # split test data in half and allocate to validation
        X_test, X_validate, y_test, y_validate = train_test_split(self.Xtest,self.ytest,test_size=0.5)
        # construct model
        model = Sequential()
        # input layer
        model.add(Dense(32,input_shape=(X_train.shape[1],)))
        # three hidden layers
        model.add(Dense(32,Activation('relu')))
        model.add(Dense(64,Activation('relu')))
        model.add(Dense(128,Activation('relu')))
        # output layer
        model.add(Dense(1))
        learning_rate = 0.001
        # stochastic gradient descent
        optimizer = optimizers.SGD(learning_rate)
        # compile - binary crossentropy for binary classification
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),optimizer=optimizer,metrics='accuracy')
        # set epochs - number of complete passes thru dataset
        epochs = 100
        # number of samples from training set to work thru before updating params
        batch_size = 32
        # fit model in history for graphing
        history = model.fit(
            X_train,
            y_train,
            batch_size = batch_size,
            epochs = epochs,
            verbose = 1,
            shuffle = True,
            steps_per_epoch= int(X_train.shape[0] / batch_size) ,
            validation_data = (X_validate, y_validate))
        # evaluate
        loss, test_acc = model.evaluate(X_test,y_test,verbose=1)
        self.tf_acc = test_acc
        # plot accuracy over epochs
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('tensor flow model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train','Cross-Validation'],loc='upper left')
        plt.show()
        # plot loss over epochs
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('tensor flow model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Train','Cross-Validation'],loc='upper right')
        plt.show()
    
    def Torch(self):
        # implementation of PyTorch neural network model, sources:
        # https://medium.com/analytics-vidhya/pytorch-for-deep-learning-binary-classification-logistic-regression-382abd97fb43
        # https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
        # uncommenting out 'device' portions invokes CUDA
        # first, standardize input - if not already done
        # scaler = StandardScaler()
        # Xtrain = scaler.fit_transform(self.Xtrain)
        # Xtest = scaler.fit_transform(self.Xtest)
        Xtrain = self.Xtrain
        Xtest = self.Xtest
        ytrain = np.asarray(self.ytrain)
        ytest = self.ytest
        # hyper parameters
        learning_rate = 0.01
        epochs = 25
        batch_size = 64
        # make training data
        class TrainData(Dataset):
            def __init__(self,x,y):
                self.x = torch.tensor(x,dtype=torch.float32)
                self.y = torch.tensor(y,dtype=torch.float32)
                self.length = self.x.shape[0]
            def __getitem__(self,idx):
                return self.x[idx], self.y[idx]
            def __len__(self):
                return self.length
        train_data = TrainData(Xtrain,ytrain)
        # make test data
        class TestData(Dataset):
            def __init__(self,x):
                self.x = x
            def __getitem__(self,idx):
                return self.x[idx]
            def __len__(self):
                return len(self.x)
        test_data = TestData(torch.FloatTensor(Xtest))       
        # DataLoaders
        train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=1)
        # define network
        class BinaryClassification(nn.Module):
            def __init__(self):
                super(BinaryClassification,self).__init__()
                self.layer1 = nn.Linear(14,64)
                self.layer2 = nn.Linear(64,64)
                self.layer_out = nn.Linear(64,1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(p=0.1)
                self.batchnorm1 = nn.BatchNorm1d(64)
                self.batchnorm2 = nn.BatchNorm1d(64)            
            # invoked automatically when class is called
            def forward(self,inputs):
                x = self.relu(self.layer1(inputs))
                x = self.batchnorm1(x)
                x = self.relu(self.layer2(x))
                x = self.batchnorm2(x)
                x = self.dropout(x)
                x = self.layer_out(x)
                return x
        # GPU
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # initialize model
        model = BinaryClassification()
        # model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        # test accuracy
        def binary_acc(y_pred,ytest):
            y_pred_tag = torch.round(torch.sigmoid(y_pred))
            correct_results_sum = (y_pred_tag==ytest).sum().float()
            acc = correct_results_sum/ytest.shape[0]
            acc = torch.round(acc*100)
            return acc
        # train model
        model.train()
        # loop through epochs
        for e in range(1,epochs+1):
            epoch_loss = 0
            epoch_acc = 0
            # inner loop for mini-batches in SGD
            for X_batch, y_batch in train_loader:
                # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # clear the last error gradient
                optimizer.zero_grad()
                # compute model output
                y_pred = model(X_batch)
                # calculate loss
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                acc = binary_acc(y_pred, y_batch.unsqueeze(1))
                # backpropagate error through model
                loss.backward()
                # update model parameters
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
        # test model
        y_pred_list = []
        model.eval()
        with torch.no_grad():
            for X_batch in test_loader:
                # X_batch = X_batch.to(device)
                y_test_pred = model(X_batch)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.numpy()) # .cpu() btwn tag and .numpy()
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        # confusion matrix and accuracy score
        self.tor_cm = confusion_matrix(ytest, y_pred_list)
        self.tor_acc = accuracy_score(ytest, y_pred_list)

    def ind_call_prediction(self,rats,prnt=False):
        # run all functions necessary for predicting stress response from individual calls
        self.merge_split_data(rats)
        self.SupportVectorMachine()
        self.DecisionTree()
        self.TensorFlow()
        self.Torch()
        if prnt:
            print('\nPredictions from ' + str(self.name) + ':')
            print('Support Vector Machine model accuracy: ' + str(self.svm_acc))
            print('Decision Tree model accuracy: ' + str(self.dt_acc))
            print('Tensor Flow model accuracy: ' + str(self.tf_acc))
            print('PyTorch model accuracy: ' + str(self.tor_acc))
            print('PyTorch confusion matrix: \n' + str(self.tor_cm))

class Cohort():
    # cluster, run call logs, and make comparison table
    def __init__(self,excel_file=None,call_log_names=[],folder_list=[]):
        self.excel_file = excel_file
        self.call_log_names = call_log_names
        self.folder_list = folder_list
        self.slices = []
        self.rats = []
        self.key = None
        self.call_logs = []
        self.table = None

    def ratify(self):
        # process information from files and create rat objects
        # first, load ratio data and cluster by change in lever pressing
        ratio_data = pd.read_excel(self.excel_file)
        dlps = ratio_data[['avg_selfad','avg_punishment']]
        kmeans = KMeans(n_clusters=2)
        clabels = kmeans.fit(dlps).predict(dlps)
        ratio_data['cluster'] = clabels
        # determine meaning of clusters and translate number to letter code
        temp = ratio_data.groupby(['cluster']).mean()
        if temp['sr'][0] > temp['sr'][1]:
            labels = ['r' if c==0 else 's' for c in clabels]
            key = '0 = stress resistance, 1 = stress sensitivity'
        else:
            labels = ['s' if c==0 else 'r' for c in clabels]
            key = '1 = stress resistance, 0 = stress sensitivity'
        self.key = key
        # create rat objects and assign id, sr, cluster, and dfs
        for ratid,label,clabel,sr in zip(ratio_data['ratid'],labels,clabels,ratio_data['sr']):
            # empty dictionary for lists of data frames - one list = one phase
            dfs = {}
            # loop through all folders
            for folder in self.folder_list:
                # folder name accesses list
                dfs[folder] = []
                # direct to excel files
                path = os.getcwd() + '\\usv_data\\' + folder
                for fname in os.listdir(path):
                    # only add data for that rat based on ID
                    if fname.startswith(ratid):
                        df = pd.read_excel(os.path.join(path,fname))
                        # add phase column to dataframe
                        df['phase'] = folder
                        # add dataframe to list keyed by phase
                        dfs[folder].append(df)
                # for every dataframe in the phase, add string and numerical label
                for df in dfs[folder]:
                    df['cluster'] = label
                    df['clabel'] = clabel
            # create rat object and add to rat list
            rat = Rat(ratid=ratid,sr=sr,cluster=label,dfs=dfs)
            self.rats.append(rat)

    def slice_list(self):
        # create list of lists for call log folders
        # i.e. convert a call log name to a list of folders so it can access the relevant data
        for name in self.call_log_names:
            temp = []
            for folder in self.folder_list:
                # call logs named by which phases to investigate
                # 'slice' is added to list if a folder name appears in the call log name
                if folder in name:
                    temp += [folder]
            self.slices.append(temp)

    def run_call_logs(self,prnt):
        for name,s in zip(self.call_log_names,self.slices):
            # create call logs with name and list of folders
            cl = Call_Log(name=name,s=s)
            # cohort rat objects run through a call log
            cl.ind_call_prediction(self.rats,prnt=prnt)
            # add call log to cohort
            self.call_logs.append(cl)
    
    def write_file(self,ofname):
        # write output file containing model accuracies across various phases
        cols = ['Support Vector Machine','Decision Tree','Tensor Flow','PyTorch']
        data=[ [0] * len(cols) for i in range(len(self.call_logs)) ]
        for i,cl in enumerate(self.call_logs):
            data[i][0] = cl.svm_acc
            data[i][1] = cl.dt_acc
            data[i][2] = cl.tf_acc
            data[i][3] = cl.tor_acc
        table = pd.DataFrame(data,index=self.call_log_names,columns=cols)
        table.to_csv(ofname)
        self.table = table
    
    def run(self,prnt=False,ofname='outputfile.csv'):
        # run all functions
        # generate rat objects
        self.ratify()
        # prepare lists of folders for call logs
        self.slice_list()
        # print what 1s and 0s mean
        if prnt:
            print('Cohort key: ' + self.key)
        # generate call log objects and run ML modules
        self.run_call_logs(prnt)
        # write output file
        self.write_file(ofname)
        # print output table
        if prnt:
            print(self.table)

# # inputs
# ef = 'ratios.xlsx'
# f = ['a','m','p1','p2']
# cl_names = ['a','m','p1','p2','p1p2','am','mp1','amp1','amp1p2']

# # create and run cohort
# C = Cohort(excel_file=ef,call_log_names=cl_names,folder_list=f)
# C.run()
# output_table = pd.DataFrame(C.table)
# output_table.index.name = 'phase'
# print(output_table)

# demonstrate with small cohort - just p1
# tensor flow learning rate 0.001, batch size 32, epochs 100
# torch learning rate 0.01, batch size 64, epochs 25
c = Cohort(excel_file='ratios.xlsx',call_log_names=['p1'],folder_list=['p1'])
c.run(prnt=True)
