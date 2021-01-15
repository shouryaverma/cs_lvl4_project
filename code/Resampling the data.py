from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wfdb
import glob
from wfdb import processing
import scipy
from scipy import *

alldata = np.empty(shape=[0, 188])
print(alldata.shape)
all_csvs = glob.glob('./mit-bih/*.csv')
for j in all_csvs:
    print('Loading ', j)
    csvrows = np.loadtxt(j, delimiter=',')
    alldata = np.append(alldata, csvrows, axis=0)
    
print(alldata.shape)

N = alldata[alldata[:,-1]==1.0]
L = alldata[alldata[:,-1]==2.0]
R = alldata[alldata[:,-1]==3.0]
V = alldata[alldata[:,-1]==4.0]
A = alldata[alldata[:,-1]==5.0]
F = alldata[alldata[:,-1]==6.0]
f = alldata[alldata[:,-1]==7.0]
I = alldata[alldata[:,-1]==8.0]

seed=42
np.random.seed(seed)
def downsample(arr, n, seed):
    downsampled = resample(arr,replace=False,n_samples=n, random_state=seed)
    return downsampled

def upsample(arr, n, seed):
    upsampled = resample(arr,replace=True,n_samples=n,random_state=seed)
    return upsampled

all_class = [N,L,R,V,A,F,f,I]
abn_class = [L,R,V,A,F,f,I]

mean_val = np.mean([len(i) for i in abn_class], dtype= int)
sampled_val = []

for i in all_class:
    if i.shape[0]> mean_val:
        i = downsample(i,mean_val,seed)
    elif i.shape[0]< mean_val:
        i = upsample(i, mean_val,seed)
    sampled_val.append(i)
    
sampled_val = np.concatenate(sampled_val)
np.random.shuffle(sampled_val)
sampled_val_all = sampled_val
