# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:13:10 2021

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

# use preprocessing code to save numpy array for each atlas
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