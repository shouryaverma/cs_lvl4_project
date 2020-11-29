import matplotlib.pyplot as plt
import numpy as np
import glob

#loading dataset
values = np.empty(shape=[0, 189])
sample_val_all = glob.glob('/nfs/sampled_val_all.csv')


for j in sample_val_all:
    print('Loading ', j)
    csvrows = np.loadtxt(j, delimiter=',')
    values = np.append(values, csvrows, axis=0)
    
print(values.shape)
    
X = values[:,:-2]
y = values[:,-2]

print(X.shape)
print(y.shape)

#running classifiers 70-30 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=48)

from sklearn.ensemble import GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier(n_estimators=10,random_state=48)
gbc_clf.fit(X_train, y_train)  
print('Gradient Boosting Accuracy')
y_pred_gbc = gbc_clf.predict(X_test)
score_gbc = gbc_clf.score(X_test,y_test)
print(score_gbc)

from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(n_estimators=10,random_state=48)
ada_clf.fit(X_train, y_train)
print('Ada Boosting Accuracy')
y_pred_ada = ada_clf.predict(X_test)
score_ada = ada_clf.score(X_test,y_test)
print(score_ada)

from sklearn.ensemble import RandomForestClassifier
rfc_clf = RandomForestClassifier(max_depth=10, random_state = 48, n_estimators=10)
rfc_clf.fit(X_train, y_train)
print('Random Forest Accuracy')
y_pred_rfc = rfc_clf.predict(X_test)
score_rfc = rfc_clf.score(X_test,y_test)
print(score_rfc)

from sklearn.naive_bayes import GaussianNB
NB_clf = GaussianNB()
NB_clf.fit(X_train,y_train)
print('Naive Bayes Accuracy')
y_pred_NB = NB_clf.predict(X_test)
score_NB = NB_clf.score(X_test,y_test)
print(score_NB)

from sklearn.neural_network import MLPClassifier
NNMLP_clf = MLPClassifier(random_state = 48, max_iter=50)
NNMLP_clf.fit(X_train, y_train)
print('NNMLP Accuracy')
y_pred_NNMLP = NNMLP_clf.predict(X_test)
score_NNMLP = NNMLP_clf.score(X_test, y_test)
print(score_NNMLP)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
spc_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
spc_clf.fit(X_train,y_train)
print('Support Vector Accuracy')
y_pred_spc = spc_clf.predict(X_test)
score_spc = spc_clf.score(X_test,y_test)
print(score_spc)

#pca and tsne
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

x_pca = PCA(n_components = 50, random_state = 48).fit_transform(X_train)
x_tsne = TSNE(n_components = 2, perplexity=50, random_state=48).fit_transform(x_pca)

plt.figure(figsize=(15,15))
scatter = plt.scatter(x_tsne[:,0],x_tsne[:,1], c= y_train)
plt.title('PCA and TSNE Clusters')
plt.legend(*scatter.legend_elements(), title='Classes', loc='upper right')
plt.savefig('PCA_TSNE.jpeg')
plt.show()

import seaborn as sns
from sklearn.metrics import *

#confusion matrix and precision scores
gbc_cf_m = confusion_matrix(y_test,y_pred_gbc)
gbc_pr_s_macro = precision_score(y_test,y_pred_gbc,average='macro')
gbc_pr_s_micro = precision_score(y_test,y_pred_gbc,average='micro')
print('Gradient Boosting macro')
print(gbc_pr_s_macro)
print('Gradient Boosting micro')
print(gbc_pr_s_micro)

ada_cf_m = confusion_matrix(y_test,y_pred_ada)
ada_pr_s_macro = precision_score(y_test,y_pred_ada,average='macro')
ada_pr_s_micro = precision_score(y_test,y_pred_ada,average='micro')
print('ADA Boosting macro')
print(ada_pr_s_macro)
print('ADA Boosting micro')
print(ada_pr_s_micro)

rfc_cf_m = confusion_matrix(y_test,y_pred_rfc)
rfc_pr_s_macro = precision_score(y_test,y_pred_rfc,average='macro')
rfc_pr_s_micro = precision_score(y_test,y_pred_rfc,average='micro')
print('Random Forest macro')
print(rfc_pr_s_macro)
print('Random Forest micro')
print(rfc_pr_s_micro)

NB_cf_m = confusion_matrix(y_test,y_pred_NB)
NB_pr_s_macro = precision_score(y_test,y_pred_NB,average='macro')
NB_pr_s_micro = precision_score(y_test,y_pred_NB,average='micro')
print('Naive Bayes macro')
print(NB_pr_s_macro)
print('Naive Bayes micro')
print(NB_pr_s_micro)

NNMLP_cf_m = confusion_matrix(y_test,y_pred_NNMLP)
NNMLP_pr_s_macro = precision_score(y_test,y_pred_NNMLP,average='macro')
NNMLP_pr_s_micro = precision_score(y_test,y_pred_NNMLP,average='micro')
print('NNMLP macro')
print(NNMLP_pr_s_macro)
print('NNMLP micro')
print(NNMLP_pr_s_micro)

spc_cf_m = confusion_matrix(y_test,y_pred_spc)
spc_pr_s_macro = precision_score(y_test,y_pred_spc,average='macro')
spc_pr_s_micro = precision_score(y_test,y_pred_spc,average='micro')
print('Support Vector macro')
print(spc_pr_s_macro)
print('Support Vector micro')
print(spc_pr_s_micro)


#plots for confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(gbc_cf_m/np.sum(gbc_cf_m),annot=True,fmt='.2%')
plt.title('GBC confusion matrix')
plt.savefig('gbc_cfm.jpeg')

plt.figure(figsize=(10,8))
sns.heatmap(ada_cf_m/np.sum(ada_cf_m),annot=True,fmt='.2%')
plt.title('ADA confusion matrix')
plt.savefig('ada_cfm.jpeg')

plt.figure(figsize=(10,8))
sns.heatmap(rfc_cf_m/np.sum(rfc_cf_m),annot=True,fmt='.2%')
plt.title('RFC confusion matrix')
plt.savefig('rfc_cfm.jpeg')

plt.figure(figsize=(10,8))
sns.heatmap(NB_cf_m/np.sum(NB_cf_m),annot=True,fmt='.2%')
plt.title('NB confusion matrix')
plt.savefig('NB_cfm.jpeg')

plt.figure(figsize=(10,8))
sns.heatmap(NNMLP_cf_m/np.sum(NNMLP_cf_m),annot=True,fmt='.2%')
plt.title('NNMLP confusion matrix')
plt.savefig('NNMLP_cfm.jpeg')

plt.figure(figsize=(10,8))
sns.heatmap(spc_cf_m/np.sum(spc_cf_m),annot=True,fmt='.2%')
plt.title('SVC confusion matrix')
plt.savefig('SVC_cfm.jpeg')











