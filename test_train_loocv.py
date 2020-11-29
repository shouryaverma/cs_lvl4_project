import numpy as np
import glob

values = np.empty(shape=[0, 189])
sample_val_all = glob.glob('/nfs/sampled_val_all.csv')

for j in sample_val_all:
    print('Loading ', j)
    csvrows = np.loadtxt(j, delimiter=',')
    values = np.append(values, csvrows, axis=0)
    
print(values.shape)

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def leave_one_patient():
    X = values[:,:-2]
    y = values[:,-2]
    groups = values[:,-1]
    lopo = LeaveOneGroupOut()
    lopo.get_n_splits(X,y,groups)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for train_index, test_index in lopo.split(X,y,groups):
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        X_train.append(X_tr)
        X_test.append(X_te)
        y_train.append(y_tr)
        y_test.append(y_te)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = leave_one_patient()

gbc_sc = []
for x1,x2,y1,y2 in X_train,X_test,y_train,y_test:

    gbc_clf = GradientBoostingClassifier(n_estimators=5)
    gbc_clf.fit(x1, y1)  
    gbc_sc.append(gbc_clf.score(x2,y2))
    
ada_sc = []
for x1,x2,y1,y2 in X_train,X_test,y_train,y_test:

    ada_clf = AdaBoostClassifier(n_estimators=5,random_state=48)
    ada_clf.fit(x1, y1)
    ada_sc.append(ada_clf.score(x2,y2))
    
rfc_sc = []    
for x1,x2,y1,y2 in X_train,X_test,y_train,y_test:

    # using random forest classifier
    rfc_clf = RandomForestClassifier(max_depth=1, random_state=48)
    rfc_clf.fit(x1, y1)
    rfc_sc.append(rfc_clf.score(x2, y2))
    
gnb_sc = []   
for x1,x2,y1,y2 in X_train,X_test,y_train,y_test:

    # using naive bayes
    NB_clf = GaussianNB()
    NB_clf.fit(x1, y1)
    gnb_sc.append(NB_clf.score(x2, y2))
    
mlp_sc = []
for x1,x2,y1,y2 in X_train,X_test,y_train,y_test:

    # using NN Multi Layer Perceptron classifier
    NNMLP_clf = MLPClassifier(random_state=48, max_iter=50)
    NNMLP_clf.fit(x1, y1)
    mlp_sc.append(NNMLP_clf.score(x2, y2))

svc_sc = []
for x1,x2,y1,y2 in X_train,X_test,y_train,y_test:

    svc_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    svc_clf.fit = clf.fit(x1, y1)
    svc_sc.append(svc_clf.score(x2,y2))

print('Gradient Boosting Results ', np.mean(gbc_sc))
print('Ada Boosting Results ', np.mean(ada_sc))
print('Random Forest Results ', np.mean(rfc_sc))
print('GaussianNB Results ', np.mean(gnb_sc))
print('NNMLP Classifier Results ', np.mean(mlp_sc))
print('Support Vector Results ', np.mean(svc_sc))