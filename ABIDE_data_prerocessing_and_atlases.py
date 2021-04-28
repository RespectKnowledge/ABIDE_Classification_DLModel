# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:10:07 2021

@author: Abdul Qayyum
"""
#%% ########################### Data preprocessing ######################

#%% ABIDE dataset to extract raw signals with time from 4D matrix into 2D matrix
import os
import warnings
from os.path import join
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from nilearn.input_data import NiftiLabelsMasker

################################################ msdl ##################################

def msdlmasker(data):
    msdl = datasets.fetch_atlas_msdl()
    #Iterate over fetched atlases to extract coordinates - probabilistic
    # create masker to extract functional data within atlas parcels
    maskermsdl = NiftiMapsMasker(maps_img=msdl['maps'], standardize=True,memory='nilearn_cache')
    timeseriesmsdl=maskermsdl.fit_transform(data)
    return timeseriesmsdl

################################################ harvardmasker ##################################

def harvardmasker(data):
    dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    #harvard_oxford_sub = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm')
    atlas_filename = dataset.maps
    labels = dataset.labels
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    # The Nifti data can then be turned to time-series by calling the NiftiLabelsMasker 
    # fit_transform method, that takes either filenames or NiftiImage objects:
    time_seriesh = masker.fit_transform(data)
    return time_seriesh
################################################ basc2015 ##################################
from nilearn.input_data import NiftiLabelsMasker
def basc2015masker444(data):
    basc = datasets.fetch_atlas_basc_multiscale_2015() # the BASC multiscale atlas
    # We use a nilearn masker to load time series from the parcellation above
    masker444 = NiftiLabelsMasker(basc['scale444'], resampling_target="data", detrend=True,
                               standardize=True, smoothing_fwhm=5, 
                               memory='nilearn_cache', memory_level=1).fit()
    tseriesbasc444 = masker444.transform(data)
    return tseriesbasc444


def basc2015masker20(data):
    basc = datasets.fetch_atlas_basc_multiscale_2015() # the BASC multiscale atlas
    # We use a nilearn masker to load time series from the parcellation above
    masker20 = NiftiLabelsMasker(basc['scale020'], resampling_target="data", detrend=True,
                               standardize=True, smoothing_fwhm=5, 
                               memory='nilearn_cache', memory_level=1).fit()
    tseriesbasc20 = masker20.transform(data)
    return tseriesbasc20

def basc2015masker64(data):
    basc = datasets.fetch_atlas_basc_multiscale_2015() # the BASC multiscale atlas
    # We use a nilearn masker to load time series from the parcellation above
    masker64 = NiftiLabelsMasker(basc['scale064'], resampling_target="data", detrend=True,
                               standardize=True, smoothing_fwhm=5, 
                               memory='nilearn_cache', memory_level=1).fit()
    tseriesbasc64 = masker64.transform(data)
    return tseriesbasc64

################################################ pauli2017masker ##################################
def pauli2017masker(data):
    subcortex = datasets.fetch_atlas_pauli_2017()
    # create masker to extract functional data within atlas parcels
    maskermsdl = NiftiMapsMasker(maps_img=subcortex['maps'], 
                                 standardize=True,
                                 memory='nilearn_cache')

    timeseriespauli=maskermsdl.fit_transform(data)
    return timeseriespauli

################################################ Smith ICA Atlas and Brain Maps 2009 ##################################
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
def smith10(data):
    smith = datasets.fetch_atlas_smith_2009()
    smith10 = NiftiMapsMasker(smith.rsn10, resampling_target="data", detrend=True,
                              standardize=True, smoothing_fwhm=5,
                              memory='nilearn_cache', memory_level=1).fit()
    tseriesbascs10= smith10.transform(data)
    return tseriesbascs10

################################################ Smith ICA Atlas and Brain Maps 2009 ###############

def smith20(data):
    smith = datasets.fetch_atlas_smith_2009()
    smith20 = NiftiMapsMasker(smith.rsn20, resampling_target="data", detrend=True,
                              standardize=True, smoothing_fwhm=5,
                              memory='nilearn_cache', memory_level=1).fit()
    tseriesbascs20= smith20.transform(data)
    return tseriesbascs20

################################################ Smith ICA Atlas and Brain Maps 2009 ###############
def smith70(data):
    smith = datasets.fetch_atlas_smith_2009()
    smith70 = NiftiMapsMasker(smith.rsn70, resampling_target="data", detrend=True,
                              standardize=True, smoothing_fwhm=5,
                              memory='nilearn_cache', memory_level=1).fit()
    tseriesbascs70= smith70.transform(data)
    return tseriesbascs70

############################### dataset loading from the abid2 csv file #############
import os
import warnings
from os.path import join
import numpy as np
import pandas as pd
import nibabel as nib

pheno_dir = 'D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\ABIDEII_dataset.csv'
pathdata='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\abdulq-20210113_061101'

datasave='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\np_msdl/'
datasave1='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\np_hardawrd/'
datasave2='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\np_basc15444/'
datasave3='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\np_basc201520/'
datasave4='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\np_basc201564/'
datasave5='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\np_pauli2017/'
datasave6='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\np_smith10/'
datasave7='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\np_smith20/'
datasave8='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\np_smith70/'

phenotypes = pd.read_csv(pheno_dir)
phenotypes
particicipant=phenotypes['participant_id']
# subids = phenotypes['SUB_ID']
# subid = phenotypes['SITE_ID']
subids=particicipant
mj_users=[]
subject_ids=[]
subject_ids_1=[]
subject_ids_2=[]
class1=[]
class2=[]
class1features=[]
class2features=[]
import shutil, os, glob
for index,sub_id in enumerate(subids):
    print(sub_id)
    #datapath=os.path.join(data_direct1+'00'+str(sub_id),"session_1"+'/'+"rest_1"+'/'+"rest.nii.gz")
    datapath=os.path.join(pathdata,sub_id+'\\'+"rest_1"+'\\'+'NIfTI-1'+'\\'+"rest"+".nii.gz")
    ######################## msdl #######################
    timeseriesmsdl=msdlmasker(data=datapath)
    np.save(os.path.join(datasave,str(sub_id)+'.npy'),timeseriesmsdl)
    ######################## harvard #######################
    time_seriesh=harvardmasker(data=datapath)
    np.save(os.path.join(datasave1,str(sub_id)+'.npy'),time_seriesh)
    ######################## basc2015masker444 #######################
    tseriesbasc444=basc2015masker444(data=datapath)
    np.save(os.path.join(datasave2,str(sub_id)+'.npy'),tseriesbasc444)
    ######################## basc2015masker20 #######################
    tseriesbasc20=basc2015masker20(data=datapath)
    np.save(os.path.join(datasave3,str(sub_id)+'.npy'),tseriesbasc20)
    ######################## basc2015masker64 #######################
    tseriesbasc64=basc2015masker64(data=datapath)
    np.save(os.path.join(datasave4,str(sub_id)+'.npy'),tseriesbasc64)
    ######################## pauli2017masker #######################
    timeseriespauli=pauli2017masker(data=datapath)
    np.save(os.path.join(datasave5,str(sub_id)+'.npy'),timeseriespauli)
         
    ######################## smith10 #######################
    tseriesbascs10=smith10(data=datapath)
    np.save(os.path.join(datasave6,str(sub_id)+'.npy'),tseriesbascs10)
        
    ######################## smith20 #######################
    tseriesbascs20=smith20(data=datapath)
    np.save(os.path.join(datasave7,str(sub_id)+'.npy'),tseriesbascs20)
        
        
    ######################## smith70 #######################
    tseriesbascs70=smith70(data=datapath)
    np.save(os.path.join(datasave8,str(sub_id)+'.npy'),tseriesbascs70)
    
    
print('work done') 
#%% DiFuMo atlases
########################### please download DiFuMo function from follwoing link
#https://github.com/Parietal-INRIA/DiFuMo/blob/master/notebook/demo.ipynb

"""Function to fetch DiFuMo atlases.
   Direct download links from OSF:
   dic = {64: https://osf.io/pqu9r/download,
          128: https://osf.io/wjvd5/download,
          256: https://osf.io/3vrct/download,
          512: https://osf.io/9b76y/download,
          1024: https://osf.io/34792/download,
          }
"""
import os
import pandas as pd

from sklearn.utils import Bunch

from nilearn.datasets.utils import (_fetch_files,
                                    _get_dataset_dir)


def fetch_difumo(dimension=64, resolution_mm=2, data_dir=None):
    """Fetch DiFuMo brain atlas
    Parameters
    ----------
    dimension : int
        Number of dimensions in the dictionary. Valid resolutions
        available are {64, 128, 256, 512, 1024}.
    resolution_mm : int
        The resolution in mm of the atlas to fetch. Valid options
        available are {2, 3}.
    data_dir : string, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory.
    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
        - 'maps': str, 4D path to nifti file containing regions definition.
        - 'labels': string list containing the labels of the regions.
    References
    ----------
    Dadi, K., Varoquaux, G., Machlouzarides-Shalit, A., Gorgolewski, KJ.,
    Wassermann, D., Thirion, B., Mensch, A.
    Fine-grain atlases of functional modes for fMRI analysis,
    Paper in preparation
    """
    dic = {64: 'pqu9r',
           128: 'wjvd5',
           256: '3vrct',
           512: '9b76y',
           1024: '34792',
           }
    valid_dimensions = [64, 128, 256, 512, 1024]
    valid_resolution_mm = [2, 3]
    if dimension not in valid_dimensions:
        raise ValueError("Requested dimension={} is not available. Valid "
                         "options: {}".format(dimension, valid_dimensions))
    if resolution_mm not in valid_resolution_mm:
        raise ValueError("Requested resolution_mm={} is not available. Valid "
                         "options: {}".format(resolution_mm,
                                              valid_resolution_mm))
    url = 'https://osf.io/{}/download'.format(dic[dimension])
    opts = {'uncompress': True}

    csv_file = os.path.join('{0}', 'labels_{0}_dictionary.csv')
    if resolution_mm != 3:
        nifti_file = os.path.join('{0}', '2mm', 'maps.nii.gz')
    else:
        nifti_file = os.path.join('{0}', '3mm', 'maps.nii.gz')

    files = [(csv_file.format(dimension), url, opts),
             (nifti_file.format(dimension), url, opts)]

    dataset_name = 'difumo_atlases'

    data_dir = _get_dataset_dir(data_dir=data_dir, dataset_name=dataset_name,
                                verbose=1)

    # Download the zip file, first
    files = _fetch_files(data_dir, files, verbose=2)
    labels = pd.read_csv(files[0])

    # README
    readme_files = [('README.md', 'https://osf.io/4k9bf/download',
                    {'move': 'README.md'})]
    if not os.path.exists(os.path.join(data_dir, 'README.md')):
        _fetch_files(data_dir, readme_files, verbose=2)

    return Bunch(maps=files[1], labels=labels)


import os
import warnings
from os.path import join
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMapsMasker

pheno_dir = 'D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\ABIDEII_dataset.csv'
pathdata='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\abdulq-20210113_061101'

datasave='D:\\Moonaradiomics\\fMRIdatasetcodes\\Abdi2dataset\\saved_numpy\\difumo/'

phenotypes = pd.read_csv(pheno_dir)
phenotypes
particicipant=phenotypes['participant_id']
# subids = phenotypes['SUB_ID']
# subid = phenotypes['SITE_ID']
subids=particicipant
mj_users=[]
subject_ids=[]
subject_ids_1=[]
subject_ids_2=[]
class1=[]
class2=[]
class1features=[]
class2features=[]
import shutil, os, glob
for index,sub_id in enumerate(subids):
    print(sub_id)
    #datapath=os.path.join(data_direct1+'00'+str(sub_id),"session_1"+'/'+"rest_1"+'/'+"rest.nii.gz")
    datapath=os.path.join(pathdata,sub_id+'\\'+"rest_1"+'\\'+'NIfTI-1'+'\\'+"rest"+".nii.gz")
    ######################## msdl #######################
    maps_img = fetch_difumo(dimension=64).maps # extract 64 bases masker
    maps_masker = NiftiMapsMasker(maps_img=maps_img, verbose=1) 
    timseriesdifumo= maps_masker.fit_transform(datapath)
    print("Per ROIs signal: {0}".format(timseriesdifumo.shape)) 
    np.save(os.path.join(datasave,str(sub_id)+'.npy'),timseriesdifumo)
      
print('work done') 