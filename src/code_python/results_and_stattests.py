# -*- coding: utf-8 -*-
"""results_and_stattests.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1X_8nVgEf7nn4B0lef-XlJ390BZSIcGuA
"""

!pip install researchpy

import numpy as np
import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt
from statistics import mean, stdev
from pylab import rcParams


sns.set()
plt.set_cmap('tab10')

# sklearn_results = pd.read_csv('./drive/MyDrive/compsci/sklearn_results.csv')
# cnn_results = pd.read_csv('./drive/MyDrive/compsci/cnn_results.csv')
# cnn_results = cnn_results.transpose()
# lstm_results = pd.read_csv('./drive/MyDrive/compsci/lstm_results.csv')
# lstm_results = lstm_results.transpose()

sklearn_results = pd.read_csv('./drive/MyDrive/compsci/leave_patients_results/macro/sklearn_results.csv')
cnn_results = pd.read_csv('./drive/MyDrive/compsci/leave_patients_results/macro/cnn_results.csv')
cnn_results = cnn_results.transpose()
lstm_results = pd.read_csv('./drive/MyDrive/compsci/leave_patients_results/macro/lstm_results.csv')
lstm_results = lstm_results.transpose()

crossval_results = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/crossval_results.csv')

sklearn_results.columns = ['Model','Accuracy','Precision','Recall','F1score','Conf Matrice']
cnn_results.columns = ['Model','Accuracy','Precision','Recall','F1score','Conf Matrice']
lstm_results.columns = ['Model','Accuracy','Precision','Recall','F1score','Conf Matrice']

crossval_results.columns = ['Model','Acc_Mean','Acc_Std','Pre_Mean','Pre_Std','Rec_Mean','Rec_Std','F1_Mean','F1_Std']

frames = [sklearn_results,lstm_results,cnn_results]
all_results = pd.concat(frames, ignore_index=True)
all_results = all_results.drop(columns=['Conf Matrice'])
print(all_results)
print('')

all_cv_results = crossval_results
print(all_cv_results.to_string())

accuracy =  round(all_results['Accuracy'].astype(float),3).to_numpy()
precision = round(all_results['Precision'].astype(float),3).to_numpy()
recall =    round(all_results['Recall'].astype(float),3).to_numpy()
f1_score =  round(all_results['F1score'].astype(float),3).to_numpy()
index = all_results['Model']
ticks = (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)

df_res = pd.DataFrame({'Precision':precision,
                   'Recall':recall,
                   'f1 Score':f1_score,
                   'Accuracy':accuracy
                  },
                  index=index)
ax = df_res.plot.bar(figsize=(14,6),
            ylim=(0,1),
            yticks=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),
            fontsize=(12),
            rot=0,
            table=True,
            title=('Weighted Average Performance Metrics per Model'),
            colormap='tab10')

h,l = ax.get_legend_handles_labels()
ax.legend(h[:4],["Precision", "Recall", "f1 Score", 'Accuracy'], loc=3, fontsize=12)
ax.axes.get_xaxis().set_visible(False)
# Getting the table created by pandas and matplotlib
table = ax.tables[0]
# Setting the font size
table.set_fontsize(12)
# Rescaling the rows to be more readable
table.scale(1,2)

import researchpy as rp
rp.summary_cont(df_res)

stack_df = df_res.stack().reset_index()
stack_df = stack_df.rename(columns={'level_0': 'models',
                                    'level_1': 'metric',
                                    0:'score'},)
display(stack_df)

plt.figure(figsize=(7,7))
mypal=('#4682B4','#DC143C','#EE82EE','#00CED1')
ax = sns.boxplot(y=stack_df["score"], x=stack_df["metric"],width=0.5, palette=mypal)
ax.set_title('Boxplot of Leave Groups Out Performance Metrics')
ax.set_ylim(0.0,1.0)
ax.set_yticks(ticks=ticks,minor=False)

accuracyM =  round(all_cv_results['Acc_Mean'].astype(float),3).to_numpy()
accuracyS =  round(all_cv_results['Acc_Std'].astype(float),3).to_numpy()

precisionM = round(all_cv_results['Pre_Mean'].astype(float),3).to_numpy()
precisionS = round(all_cv_results['Pre_Std'].astype(float),3).to_numpy()

recallM =    round(all_cv_results['Rec_Mean'].astype(float),3).to_numpy()
recallS =    round(all_cv_results['Rec_Std'].astype(float),3).to_numpy()

f1_scoreM =  round(all_cv_results['F1_Mean'].astype(float),3).to_numpy()
f1_scoreS =  round(all_cv_results['F1_Std'].astype(float),3).to_numpy()

index = all_cv_results['Model']
ticks = (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)

df_cv = pd.DataFrame({'Precision Mean':precisionM,
                       'Recall Mean':recallM,
                       'F1_score Mean':f1_scoreM,
                       'Accuracy Mean':accuracyM,
                      # 'Precision Std':precisionS,
                      # 'Recall Std':recallS,                   
                      # 'F1_score Std':f1_scoreS,                   
                      # 'Accuracy Std':accuracyS
                       },index=index)

error = [precisionS,recallS,f1_scoreS,accuracyS]

ax = df_cv.plot.bar(figsize=(14,6),
            ylim=(0,1),
            yticks=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),
            fontsize=(12),
            rot=0,
            yerr=error,
            capsize=5,
            table=True,
            title=('Macro Average Performance Metrics for Cross Validation per Model'),
            colormap='tab10')

h,l = ax.get_legend_handles_labels()
ax.legend(h[:4],["Precision", "Recall", "f1 Score", 'Accuracy'], loc=4, fontsize=12)
ax.axes.get_xaxis().set_visible(False)
# Getting the table created by pandas and matplotlib
table = ax.tables[0]
# Setting the font size
table.set_fontsize(12)
# Rescaling the rows to be more readable
table.scale(1,2)

import researchpy as rp
rp.summary_cont(df_cv)

stack_df = df_cv.stack().reset_index()
stack_df = stack_df.rename(columns={'level_0': 'models',
                                    'level_1': 'metric',
                                    0:'score'},)
display(stack_df)

plt.figure(figsize=(7,7))
mypal=('#4682B4','#DC143C','#EE82EE','#00CED1')
ax = sns.boxplot(y=stack_df["score"], x=stack_df["metric"],width=0.5, palette=mypal)
ax.set_title('Boxplot of KFold Cross Validation Performance Metrics')
ax.set_ylim(0.0,1.0)
ax.set_yticks(ticks=ticks,minor=False)

import numpy as np
import scipy.stats as stats
# to check
# Confidence interval
confidence_level = 0.95
# If juste samples (not mean) the dof is 
mean_list, std_list, ci_list = [], [], []
for col_name  in df_cv:
    col_values = df_cv[col_name].values
    sample_size = len(col_values)
    degrees_freedom = sample_size - 1
    sample_mean = np.mean(col_values)
    # Standard error of the mean (SEM) = sigma / sqrt(n)
    sample_standard_error = stats.sem(col_values)
    print('sample_standard_error s^2=', sample_standard_error,
         'or s/np.sqrt(n_t)', np.std(col_values)/np.sqrt(sample_size),  np.std(col_values))
    confidence_interval = stats.t.interval(alpha=confidence_level,
                                           df=degrees_freedom,
                                           loc=sample_mean,
                                           scale=sample_standard_error)
    std_list.append(sample_standard_error)
    ci_list.append(confidence_interval)
    mean_list.append(sample_mean)
    
CI_df = pd.DataFrame([df_cv.columns.values, mean_list, std_list,  ci_list]).transpose()
CI_df.columns = ['metric',
                 'mean',
                 'std error',
                 'CI']
CI_df.loc[:,'CI'] =  CI_df.loc[:,'CI'].map(lambda x: (x[0].round(2), x[1].round(2)))
CI_df = CI_df.sort_values(by=['mean'])
display(CI_df)

mypal1=('#4682B4','#DC143C','#EE82EE','#00CED1')
graph = sns.displot(stack_df, x='score', hue='metric', kind="kde", fill=True,height=5,aspect=2,legend=False,palette=mypal1)
plt.title('Distplot of Kfold Cross Validation Performance Metrics')
plt.ylim(0,1.5)
plt.xlabel('Confidence Interval')
graph.ax.legend(labels=['Accuracy','Recall','Precision','f1 Score'],loc=2)
for CI in CI_df['CI'].values:
    plt.axvline(CI[0],  linestyle='--')
plt.show()

ADA_acc = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/ADA_acc.csv') 
ADA_pre = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/ADA_pre.csv') 
ADA_rec = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/ADA_rec.csv') 
ADA_f1s = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/ADA_f1s.csv') 

RFC_acc = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/RFC_acc.csv') 
RFC_pre = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/RFC_pre.csv') 
RFC_rec = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/RFC_rec.csv') 
RFC_f1s = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/RFC_f1s.csv') 

NB_acc = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/NB_acc.csv') 
NB_pre = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/NB_pre.csv') 
NB_rec = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/NB_rec.csv') 
NB_f1s = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/NB_f1s.csv') 

NNMLP_acc = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/NNMLP_acc.csv') 
NNMLP_pre = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/NNMLP_pre.csv') 
NNMLP_rec = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/NNMLP_rec.csv') 
NNMLP_f1s = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/NNMLP_f1s.csv') 

SVC_acc = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/SVC_acc.csv') 
SVC_pre = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/SVC_pre.csv') 
SVC_rec = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/SVC_rec.csv') 
SVC_f1s = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/SVC_f1s.csv') 

LSTM_acc = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/LSTM_acc.csv') 
LSTM_pre = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/LSTM_pre.csv') 
LSTM_rec = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/LSTM_rec.csv') 
LSTM_f1s = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/LSTM_f1s.csv') 

CNN_acc = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/CNN_acc.csv') 
CNN_pre = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/CNN_pre.csv') 
CNN_rec = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/CNN_rec.csv') 
CNN_f1s = pd.read_csv('./drive/MyDrive/compsci/metric_results_macro/CNN_f1s.csv')

from scipy.stats import *
norm_data = [ADA_acc,ADA_pre,ADA_rec,ADA_f1s,
             RFC_acc,RFC_pre,RFC_rec,RFC_f1s,
             NB_acc,NB_pre,NB_rec,NB_f1s,
             NNMLP_acc,NNMLP_pre,NNMLP_rec,NNMLP_f1s,
             SVC_acc,SVC_pre,SVC_rec,SVC_f1s,
             LSTM_acc,LSTM_pre,LSTM_rec,LSTM_f1s,
             CNN_acc,CNN_pre,CNN_rec,CNN_f1s]

for i in norm_data:
  print(shapiro(i))

from scipy.stats import *

# Perform one-way ANOVA.

f_acc_stat, f_acc_p = f_oneway(ADA_acc,RFC_acc,NB_acc,NNMLP_acc,SVC_acc,LSTM_acc,CNN_acc)
f_pre_stat, f_pre_p = f_oneway(ADA_pre,RFC_pre,NB_pre,NNMLP_pre,SVC_pre,LSTM_pre,CNN_pre)
f_rec_stat, f_rec_p = f_oneway(ADA_rec,RFC_rec,NB_rec,NNMLP_rec,SVC_rec,LSTM_rec,CNN_rec)
f_f1s_stat, f_f1s_p = f_oneway(ADA_f1s,RFC_f1s,NB_f1s,NNMLP_f1s,SVC_f1s,LSTM_f1s,CNN_f1s)

print('Accuracy: ', f_oneway(ADA_acc,RFC_acc,NB_acc,NNMLP_acc,SVC_acc,LSTM_acc,CNN_acc))
print('Precision: ', f_oneway(ADA_pre,RFC_pre,NB_pre,NNMLP_pre,SVC_pre,LSTM_pre,CNN_pre))
print('Recall: ', f_oneway(ADA_rec,RFC_rec,NB_rec,NNMLP_rec,SVC_rec,LSTM_rec,CNN_rec))
print('F1Score: ', f_oneway(ADA_f1s,RFC_f1s,NB_f1s,NNMLP_f1s,SVC_f1s,LSTM_f1s,CNN_f1s))

k_acc_stat, k_acc_p = kruskal(ADA_acc,RFC_acc,NB_acc,NNMLP_acc,SVC_acc,LSTM_acc,CNN_acc)
k_pre_stat, k_pre_p = kruskal(ADA_pre,RFC_pre,NB_pre,NNMLP_pre,SVC_pre,LSTM_pre,CNN_pre)
k_rec_stat, k_rec_p = kruskal(ADA_rec,RFC_rec,NB_rec,NNMLP_rec,SVC_rec,LSTM_rec,CNN_rec)
k_f1s_stat, k_f1s_p = kruskal(ADA_f1s,RFC_f1s,NB_f1s,NNMLP_f1s,SVC_f1s,LSTM_f1s,CNN_f1s)

print('Accuracy: ', kruskal(ADA_acc,RFC_acc,NB_acc,NNMLP_acc,SVC_acc,LSTM_acc,CNN_acc))
print('Precision: ', kruskal(ADA_pre,RFC_pre,NB_pre,NNMLP_pre,SVC_pre,LSTM_pre,CNN_pre))
print('Recall: ', kruskal(ADA_rec,RFC_rec,NB_rec,NNMLP_rec,SVC_rec,LSTM_rec,CNN_rec))
print('F1Score: ', kruskal(ADA_f1s,RFC_f1s,NB_f1s,NNMLP_f1s,SVC_f1s,LSTM_f1s,CNN_f1s))

!pip install scikit-posthocs

from scikit_posthocs import posthoc_wilcoxon

acc_data = [ADA_acc.to_numpy(),RFC_acc.to_numpy(),NB_acc.to_numpy(),NNMLP_acc.to_numpy(),SVC_acc.to_numpy(),LSTM_acc.to_numpy(),CNN_acc.to_numpy()]

pre_data = [ADA_pre.to_numpy(),RFC_pre.to_numpy(),NB_pre.to_numpy(),NNMLP_pre.to_numpy(),SVC_pre.to_numpy(),LSTM_pre.to_numpy(),CNN_pre.to_numpy()]

rec_data = [ADA_rec.to_numpy(),RFC_rec.to_numpy(),NB_rec.to_numpy(),NNMLP_rec.to_numpy(),SVC_rec.to_numpy(),LSTM_rec.to_numpy(),CNN_rec.to_numpy()]

f1s_data = [ADA_f1s.to_numpy(),RFC_f1s.to_numpy(),NB_f1s.to_numpy(),NNMLP_f1s.to_numpy(),SVC_f1s.to_numpy(),LSTM_f1s.to_numpy(),CNN_f1s.to_numpy()]

acc_pair_wilcox = posthoc_wilcoxon(acc_data, p_adjust='bonferroni')
pre_pair_wilcox = posthoc_wilcoxon(pre_data, p_adjust='bonferroni')
rec_pair_wilcox = posthoc_wilcoxon(rec_data, p_adjust='bonferroni')
f1s_pair_wilcox = posthoc_wilcoxon(f1s_data, p_adjust='bonferroni')

print('Accuracy:\n',acc_pair_wilcox)
print('Precision:\n',pre_pair_wilcox)
print('Recall:\n',rec_pair_wilcox)
print('F1Score:\n',f1s_pair_wilcox)

categories = ['ADA','RFC','NB','NNMLP','SVC','LSTM','CNN']
rcParams['figure.figsize'] = 8,6
sns.heatmap(acc_pair_wilcox, annot=True, fmt='.2g' , xticklabels=categories,yticklabels=categories, vmin=-0.5, vmax=1)
plt.title('Accuracy Metrics Pairwise Wilcoxon, Bonferroni Correction')
plt.show()
sns.heatmap(pre_pair_wilcox, annot=True, fmt='.2g' , xticklabels=categories,yticklabels=categories, vmin=-0.5, vmax=1)
plt.title('Precision Metrics Pairwise Wilcoxon, Bonferroni Correction')
plt.show()
sns.heatmap(rec_pair_wilcox, annot=True, fmt='.2g' , xticklabels=categories,yticklabels=categories, vmin=-0.5, vmax=1)
plt.title('Recall Metrics Pairwise Wilcoxon, Bonferroni Correction')
plt.show()
sns.heatmap(f1s_pair_wilcox, annot=True, fmt='.2g' , xticklabels=categories,yticklabels=categories, vmin=-0.5, vmax=1)
plt.title('F1Score Metrics Pairwise Wilcoxon, Bonferroni Correction')
plt.show()

acc_pair_wilcox = posthoc_wilcoxon(acc_data)
pre_pair_wilcox = posthoc_wilcoxon(pre_data)
rec_pair_wilcox = posthoc_wilcoxon(rec_data)
f1s_pair_wilcox = posthoc_wilcoxon(f1s_data)

print('Accuracy:\n',acc_pair_wilcox)
print('Precision:\n',pre_pair_wilcox)
print('Recall:\n',rec_pair_wilcox)
print('F1Score:\n',f1s_pair_wilcox)

categories = ['ADA','RFC','NB','NNMLP','SVC','LSTM','CNN']
rcParams['figure.figsize'] = 8,6
sns.heatmap(acc_pair_wilcox, annot=True, fmt='.2g' , xticklabels=categories,yticklabels=categories, vmin=-0.5, vmax=1)
plt.title('Accuracy Metrics Pairwise Wilcoxon')
plt.show()
sns.heatmap(pre_pair_wilcox, annot=True, fmt='.2g' , xticklabels=categories,yticklabels=categories, vmin=-0.5, vmax=1)
plt.title('Precision Metrics Pairwise Wilcoxon')
plt.show()
sns.heatmap(rec_pair_wilcox, annot=True, fmt='.2g' , xticklabels=categories,yticklabels=categories, vmin=-0.5, vmax=1)
plt.title('Recall Metrics Pairwise Wilcoxon')
plt.show()
sns.heatmap(f1s_pair_wilcox, annot=True, fmt='.2g' , xticklabels=categories,yticklabels=categories, vmin=-0.5, vmax=1)
plt.title('F1Score Metrics Pairwise Wilcoxon')
plt.show()

