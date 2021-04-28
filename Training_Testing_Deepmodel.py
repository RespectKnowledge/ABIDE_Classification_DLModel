# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:16:02 2021

@author: Abdul Qayyum
"""

#%% dataset for classification 1DCNN+LSTM model
#%dataset class for training and testing the model
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset
import os
import pandas as pd
import nibabel as nib
import os
import warnings
from os.path import join
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiLabelsMasker


pheno_dir = 'D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\ABIDEII_dataset.csv'
data_direct1='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy/'
phenotypes = pd.read_csv(pheno_dir)
phenotypes
subids = phenotypes['participant_id'].tolist()
class_values=particicipant=phenotypes['DX_GROUP1'].tolist()
pathdata=data_direct1+"np_msdl"+"\\"+str(subids[0])+'.npy'
npload=np.load(pathdata)
# datapath=os.path.join(pathdata,"session_1"+'/'+"rest_1"+'/'+"rest.nii.gz")
# ff=nib.load(datapath)

class ABIDEdata(Dataset):
    def __init__(self,rootpath,csvfile,time_step,transform=None,masking='msdl'):
        super().__init__()
        self.rootpath=rootpath
        self.csvfile=csvfile
        self.transform=transform
        self.masking=masking
        self.time_step=time_step
        #self.msdlmasker(data)
        self.phenotypes = pd.read_csv(self.csvfile)
        self.labels=self.phenotypes['DX_GROUP1'].tolist()
        self.subjectlist=self.phenotypes['participant_id'].tolist()
    
    def __getitem__(self,idx):
        
        #################### extract time series features using msdl masking #####
        if self.masking=='msdl':
            pathdata=self.rootpath+"np_msdl"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesmsdl=np.load(pathdata)
            return timeseriesmsdl,class_labels
        
        #################### extract time series features using harvardmaske masking #####
        elif self.masking=='harvardmasker':
            pathdata=self.rootpath+"np_hardawrd"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesharvad=np.load(pathdata)
            print(class_labels)
            print("okkkkkkkkkkkkkkk")
            return timeseriesharvad,class_labels
        
        #################### extract time series features using basc2015masker444 masking #####
        
        elif self.masking=='basc2015masker444':
            pathdata=self.rootpath+"np_basc15444"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesb444=np.load(pathdata)
            print(class_labels)
            return timeseriesb444,class_labels
        
        #################### extract time series features using basc2015masker20 masking #####
        
        elif self.masking=='basc2015masker20':
            pathdata=self.rootpath+"np_basc201520"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesmsk20=np.load(pathdata)
            print(class_labels)
            return timeseriesmsk20,class_labels
        
        #################### extract time series features using basc2015masker64 masking #####
        
        elif self.masking=='basc2015masker64':
            pathdata=self.rootpath+"np_basc201564"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesmsk64=np.load(pathdata)
            print(class_labels)
            return timeseriesmsk64,class_labels
        
        #################### extract time series features using pauli2017maske masking #####
        
        elif self.masking=='pauli2017masker':
            pathdata=self.rootpath+"np_pauli2017"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesmsdl=np.load(pathdata)
            print(class_labels)
            return timeseriesmsdl,class_labels
        
        #################### extract time series features using smith10 masking #####
        
        elif self.masking=='smith10':
            pathdata=self.rootpath+'np_smith10'+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriess10=np.load(pathdata)
            print(class_labels)
            return timeseriess10,class_labels
        
        #################### extract time series features using smith20 masking #####
        
        elif self.masking=='smith20':
            pathdata=self.rootpath+'np_smith20'+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriess20=np.load(pathdata)
            timeseriess20=timeseriess20[0:self.time_step,:] # extract equal time_stamps for all features
            print(class_labels)
            return timeseriess20,class_labels
        #################### extract time series features using smith70 masking #####
        
        elif self.masking=='smith70':
            pathdata=self.rootpath+'np_smith70'+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriess70=np.load(pathdata)
            timeseriess70=timeseriess70[0:self.time_step,:] # extract equal time_stamps for all features
            timeseriess70=np.transpose(timeseriess70)
            # print(class_labels)
            #print("oksmith")
            return timeseriess70,class_labels
        elif self.masking=='difumo':
            pathdata=self.rootpath+'difumo'+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            difumo=np.load(pathdata)
            difumo=difumo[0:self.time_step,:] # extract equal time samples
            difumo=np.transpose(difumo)
            # print(class_labels)
            #print("okdifumo")
            return difumo,class_labels
        else:
            ("do nothing")
        
    def __len__(self):
        return len(self.labels)
from torchvision import transforms
#simple_transform = transforms.Compose([transforms.ToTensor()])
dataset=ABIDEdata(data_direct1,pheno_dir,time_step=120,masking="smith70")
#dataset_trans=simple_transform(dataset)
# print(len(dataset))
# img,mask=dataset[0]
# for idx,(feature,labels) in enumerate(dataset):
#     print(feature.shape)
#     print(labels)
#     print(idx)
    
n_train_examples = int(len(dataset)*0.9)
n_valid_examples = len(dataset) - n_train_examples

train_data, valid_data = torch.utils.data.random_split(dataset, 
                                                       [n_train_examples, n_valid_examples])


# print("Number of training examples:", len(train_data))
# print("Number of validation examples:", len(valid_data))


BATCH_SIZE = 4
BatchSize=BATCH_SIZE
train_loader= torch.utils.data.DataLoader(train_data, 
                                             shuffle = True, 
                                             batch_size = BATCH_SIZE)

# img,label=next(iter(train_dataloader))
# print(img.shape)
# print(label.shape)

test_loader= torch.utils.data.DataLoader(valid_data, 
                                             batch_size = BATCH_SIZE)

# img,label=next(iter(valid_dataloader))
# print(img.shape)
# print(label.shape) 

# define proposed model

import torch
import torch.nn as nn
class CNNLSTM(nn.Module):
  def __init__(self,input_size,number_layers,hidden_units,number_classes=2):
    super().__init__()
    self.number_layers=number_layers
    self.hidden_units=hidden_units
    self.input_size=input_size
    self.c1=nn.Conv1d(in_channels=input_size,out_channels=10,kernel_size=5) # 120+1-5=116
    self.b1=nn.BatchNorm1d(10)
    self.mp1=nn.MaxPool1d(2) #116/2=58
    self.c2=nn.Conv1d(10,out_channels=20,kernel_size=3) #58+1-3=56
    self.b2=nn.BatchNorm1d(20)
    self.mp2=nn.MaxPool1d(2) #56/2=28
    # b*20*30
    self.lstm=nn.LSTM(28,self.hidden_units,self.number_layers,batch_first=True)
    self.fc=nn.Linear(self.hidden_units,number_classes)

  def forward(self,x):
    x=self.mp1(self.b1(self.c1(x)))
    x=self.mp2(self.b2(self.c2(x)))
    #print(x.shape)
    # h0=torch.zeros(self.number_layers,x.size(0),self.hidden_units)
    # c0=torch.zeros(self.number_layers,x.size(0),self.hidden_units)
    # h0.cuda()
    # c0.cuda()
    # out,(h1,c1)=self.lstm(x,(h0,c0))
    out,(h1,c1)=self.lstm(x)
    out1=self.fc(out[:,-1,:]) # batch_size*hidden_units
    #print(out1.shape)
    return out1
model=CNNLSTM(70,2,64) # inp,number_layers,hidden_dim
# inpt=torch.randn(10,64,120) # batch_sizexfeatures or input channels xseq_len in time 
# out=model(inpt)   

# Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))
#model=model.to(device)
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#Define loss function and optimizer
#criterion = nn.NLLLoss() # Negative Log-likelihood
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Adam

trainLoss = [] # train loss for all epochs
trainAcc = [] # train accuracy for all epoch
testLoss = [] # test loss for all epochs
testAcc = []  # test accuracy for all epoch
minibatch_loss_list=[] #
epochs=100
logging_interval=10
#start = time.time()
for epoch in range(epochs):
    
    # model in training mode
    model.train
    # epoch start time
    #epochstart=time.time()
    # check loss and correct samples for training samples
    batchloss=0.0
    runningcorrect=0
    for batch_idx,(features,labels) in enumerate(train_loader):
        features=features.cuda()
        labels=labels.cuda()
        model=model.cuda()
        # compute forward
        outputs=model(features)
        _,predicted_labels=torch.max(outputs.data,1)
        
        # compute the correct labels based on the prediction
        runningcorrect +=(predicted_labels==labels.data).sum() # compute correct and sum across the batch
        # clear the gradient
        optimizer.zero_grad()
        # compute error between GT and prediction
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        # loss backward and compute the gradient
        loss.backward()
        # update the gradient and parameters
        optimizer.step()
        # accumulate the loss per batch
        batchloss +=loss.item()
        # store mini-batch_loss list
        minibatch_loss_list.append(loss.item()) # store minibatch_loss list
        
        if not batch_idx % logging_interval:
                print(f"Epoch: {epoch+1:03d}/{epochs:03d},Batch: {batch_idx:04d}/{len(train_loader):04d},Loss: {loss:.4f}")
    
    # compute the accuracy and loss per batch
    batchavgtrainacu=100*float(runningcorrect)/float(len(train_loader.dataset))
    batchavgtrainloss=batchloss/(float(len(train_loader))/BatchSize)
    trainAcc.append(batchavgtrainacu) # avreage accuracy across mini-batch
    trainLoss.append(batchavgtrainloss) # average loss across mini-batch
    # printing training accuracy for each epoch
    #print(f"Epoch: {epoch+1:03d}/{epochs:03d},Train:{trainAcc:0.2f}")
    
    
    # for testing or validation computation for every epoch
    torch.backends.cudnn.enabled = False
    model.eval()
    batch_correct_valid=0
    with torch.no_grad():
        for ii,(features,labels) in enumerate(test_loader):
            features=features.cuda()
            labels=labels.cuda()
            outputs=model(features)
            _,predictedvalid=torch.max(outputs.data,1)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            batchloss +=loss.item()
            batch_correct_valid +=(predictedvalid==labels.data).sum()
        
    # compute loss and other stuff
    avgTestLoss=batchloss/(float(len(test_loader.dataset))/BatchSize)
    avgTestAcc=100*float(batch_correct_valid)/float(len(test_loader.dataset))
    testAcc.append(avgTestAcc) # append test accuracy for each epoch
    testLoss.append(avgTestLoss) # append test loss for each epoch
    #printing testing accuracy for each epcoh
    #print(f"Epoch: {epoch+1:03d}/{epochs:03d},Test:{testAcc:0.2f}")
# import matplotlib.pyplot as plt
# plt.plot(trainLoss) 
# tt=np.array(testLoss)/6
plt.plot(trainAcc)
plt.plot(testAcc)
# #plt.plot(testLoss) 

import matplotlib.pyplot as plt
plt.plot(trainLoss) 
plt.plot(testLoss)   
#%
#%% dataset for classification 1DCNN
#%dataset class for training and testing the model
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset
import os
import pandas as pd
import nibabel as nib
import os
import warnings
from os.path import join
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiLabelsMasker


pheno_dir = 'D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\ABIDEII_dataset.csv'
data_direct1='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy/'
phenotypes = pd.read_csv(pheno_dir)
phenotypes
subids = phenotypes['participant_id'].tolist()
class_values=particicipant=phenotypes['DX_GROUP1'].tolist()
pathdata=data_direct1+"np_msdl"+"\\"+str(subids[0])+'.npy'
npload=np.load(pathdata)
# datapath=os.path.join(pathdata,"session_1"+'/'+"rest_1"+'/'+"rest.nii.gz")
# ff=nib.load(datapath)

class ABIDEdata(Dataset):
    def __init__(self,rootpath,csvfile,time_step,transform=None,masking='msdl'):
        super().__init__()
        self.rootpath=rootpath
        self.csvfile=csvfile
        self.transform=transform
        self.masking=masking
        self.time_step=time_step
        #self.msdlmasker(data)
        self.phenotypes = pd.read_csv(self.csvfile)
        self.labels=self.phenotypes['DX_GROUP1'].tolist()
        self.subjectlist=self.phenotypes['participant_id'].tolist()
    
    def __getitem__(self,idx):
        
        #################### extract time series features using msdl masking #####
        if self.masking=='msdl':
            pathdata=self.rootpath+"np_msdl"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesmsdl=np.load(pathdata)
            return timeseriesmsdl,class_labels
        
        #################### extract time series features using harvardmaske masking #####
        elif self.masking=='harvardmasker':
            pathdata=self.rootpath+"np_hardawrd"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesharvad=np.load(pathdata)
            print(class_labels)
            print("okkkkkkkkkkkkkkk")
            return timeseriesharvad,class_labels
        
        #################### extract time series features using basc2015masker444 masking #####
        
        elif self.masking=='basc2015masker444':
            pathdata=self.rootpath+"np_basc15444"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesb444=np.load(pathdata)
            print(class_labels)
            return timeseriesb444,class_labels
        
        #################### extract time series features using basc2015masker20 masking #####
        
        elif self.masking=='basc2015masker20':
            pathdata=self.rootpath+"np_basc201520"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesmsk20=np.load(pathdata)
            print(class_labels)
            return timeseriesmsk20,class_labels
        
        #################### extract time series features using basc2015masker64 masking #####
        
        elif self.masking=='basc2015masker64':
            pathdata=self.rootpath+"np_basc201564"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesmsk64=np.load(pathdata)
            print(class_labels)
            return timeseriesmsk64,class_labels
        
        #################### extract time series features using pauli2017maske masking #####
        
        elif self.masking=='pauli2017masker':
            pathdata=self.rootpath+"np_pauli2017"+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriesmsdl=np.load(pathdata)
            print(class_labels)
            return timeseriesmsdl,class_labels
        
        #################### extract time series features using smith10 masking #####
        
        elif self.masking=='smith10':
            pathdata=self.rootpath+'np_smith10'+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriess10=np.load(pathdata)
            print(class_labels)
            return timeseriess10,class_labels
        
        #################### extract time series features using smith20 masking #####
        
        elif self.masking=='smith20':
            pathdata=self.rootpath+'np_smith20'+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriess20=np.load(pathdata)
            timeseriess20=timeseriess20[0:self.time_step,:] # extract equal time_stamps for all features
            print(class_labels)
            return timeseriess20,class_labels
        #################### extract time series features using smith70 masking #####
        
        elif self.masking=='smith70':
            pathdata=self.rootpath+'np_smith70'+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            timeseriess70=np.load(pathdata)
            timeseriess70=timeseriess70[0:self.time_step,:] # extract equal time_stamps for all features
            # print(class_labels)
            print("oksmith")
            return timeseriess70,class_labels
        elif self.masking=='difumo':
            pathdata=self.rootpath+'difumo'+"\\"+str(self.subjectlist[idx])+'.npy'
            class_labels=self.labels[idx]
            difumo=np.load(pathdata)
            difumo=difumo[0:self.time_step,:] # extract equal time samples
            difumo=np.transpose(difumo)
            # print(class_labels)
            #print("okdifumo")
            return difumo,class_labels
        else:
            ("do nothing")
        
    def __len__(self):
        return len(self.labels)
from torchvision import transforms
#simple_transform = transforms.Compose([transforms.ToTensor()])
dataset=ABIDEdata(data_direct1,pheno_dir,time_step=120,masking="difumo")
#dataset_trans=simple_transform(dataset)
# print(len(dataset))
# img,mask=dataset[0]
# for idx,(feature,labels) in enumerate(dataset):
#     print(feature.shape)
#     print(labels)
#     print(idx)
    
n_train_examples = int(len(dataset)*0.9)
n_valid_examples = len(dataset) - n_train_examples

train_data, valid_data = torch.utils.data.random_split(dataset, 
                                                       [n_train_examples, n_valid_examples])


# print("Number of training examples:", len(train_data))
# print("Number of validation examples:", len(valid_data))


BATCH_SIZE = 4
BatchSize=BATCH_SIZE
train_loader= torch.utils.data.DataLoader(train_data, 
                                             shuffle = True, 
                                             batch_size = BATCH_SIZE)

# img,label=next(iter(train_dataloader))
# print(img.shape)
# print(label.shape)

test_loader= torch.utils.data.DataLoader(valid_data, 
                                             batch_size = BATCH_SIZE)

# img,label=next(iter(valid_dataloader))
# print(img.shape)
# print(label.shape) 

# define proposed model

import torch
import torch.nn as nn
# 1D CNN model in pytorch using some layer combinations
import torch.nn as nn
class CNN_model(nn.Module):
  def __init__(self,in_chan,classes):
    super(CNN_model,self).__init__()
    # block 1
    self.c11=nn.Conv1d(in_channels=in_chan,out_channels=120,kernel_size=5) # 120-5+1=116
    self.maxpool11=nn.MaxPool1d(2) #58
    self.c21=nn.Conv1d(in_channels=120,out_channels=130,kernel_size=3) #58-5+1=56
    self.maxpool21=nn.MaxPool1d(2) #28
    # block2
    self.c12=nn.Conv1d(in_channels=130,out_channels=140,kernel_size=3) #28-1+1
    self.maxpool12=nn.MaxPool1d(2) #15/3=13
    self.c22=nn.Conv1d(in_channels=140,out_channels=150,kernel_size=3) #13-3+1 
    # linear layer
    self.fc1=nn.Linear(150*11,60)
    self.fc2=nn.Linear(60,classes)

  def forward(self,x):
    # block 1 
    x=self.maxpool21(self.c21(self.maxpool11(self.c11(x))))
    
    x=(self.c22(self.maxpool12(self.c12(x))))
    x=x.view(-1,150*11)
    x=self.fc1(x)
    x=self.fc2(x)
    return x
model=CNN_model(in_chan=64,classes=2) # inp,number_layers,hidden_dim
# inpt=torch.randn(10,64,120) # batch_sizexfeatures or input channels xseq_len in time 
# out=model(inpt)   

# Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))
#model=model.to(device)
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#Define loss function and optimizer
#criterion = nn.NLLLoss() # Negative Log-likelihood
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Adam

trainLoss = [] # train loss for all epochs
trainAcc = [] # train accuracy for all epoch
testLoss = [] # test loss for all epochs
testAcc = []  # test accuracy for all epoch
minibatch_loss_list=[] #
epochs=20
logging_interval=10
#start = time.time()
for epoch in range(epochs):
    
    # model in training mode
    model.train
    # epoch start time
    #epochstart=time.time()
    # check loss and correct samples for training samples
    batchloss=0.0
    runningcorrect=0
    for batch_idx,(features,labels) in enumerate(train_loader):
        features=features.cuda()
        labels=labels.cuda()
        model=model.cuda()
        # compute forward
        outputs=model(features)
        _,predicted_labels=torch.max(outputs.data,1)
        
        # compute the correct labels based on the prediction
        runningcorrect +=(predicted_labels==labels.data).sum() # compute correct and sum across the batch
        # clear the gradient
        optimizer.zero_grad()
        # compute error between GT and prediction
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        # loss backward and compute the gradient
        loss.backward()
        # update the gradient and parameters
        optimizer.step()
        # accumulate the loss per batch
        batchloss +=loss.item()
        # store mini-batch_loss list
        minibatch_loss_list.append(loss.item()) # store minibatch_loss list
        
        if not batch_idx % logging_interval:
                print(f"Epoch: {epoch+1:03d}/{epochs:03d},Batch: {batch_idx:04d}/{len(train_loader):04d},Loss: {loss:.4f}")
    
    # compute the accuracy and loss per batch
    batchavgtrainacu=100*float(runningcorrect)/float(len(train_loader.dataset))
    batchavgtrainloss=batchloss/(float(len(train_loader))/BatchSize)
    trainAcc.append(batchavgtrainacu) # avreage accuracy across mini-batch
    trainLoss.append(batchavgtrainloss) # average loss across mini-batch
    # printing training accuracy for each epoch
    #print(f"Epoch: {epoch+1:03d}/{epochs:03d},Train:{trainAcc:0.2f}")
    
    
    # for testing or validation computation for every epoch
    torch.backends.cudnn.enabled = False
    model.eval()
    batch_correct_valid=0
    with torch.no_grad():
        for ii,(features,labels) in enumerate(test_loader):
            features=features.cuda()
            labels=labels.cuda()
            outputs=model(features)
            _,predictedvalid=torch.max(outputs.data,1)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            batchloss +=loss.item()
            batch_correct_valid +=(predictedvalid==labels.data).sum()
        
    # compute loss and other stuff
    avgTestLoss=batchloss/(float(len(test_loader.dataset))/BatchSize)
    avgTestAcc=100*float(batch_correct_valid)/float(len(test_loader.dataset))
    testAcc.append(avgTestAcc) # append test accuracy for each epoch
    testLoss.append(avgTestLoss) # append test loss for each epoch
    #printing testing accuracy for each epcoh
    #print(f"Epoch: {epoch+1:03d}/{epochs:03d},Test:{testAcc:0.2f}")
# import matplotlib.pyplot as plt
# plt.plot(trainLoss) 
# #tt=np.array(testLoss)/6
# #plt.plot(trainAcc)
# #plt.plot(testAcc)
# plt.plot(testLoss) 

# import matplotlib.pyplot as plt
# plt.plot(trainLoss) 
# plt.plot(testLoss)   

# plt.plot(trainAcc)
# plt.plot(testAcc)
