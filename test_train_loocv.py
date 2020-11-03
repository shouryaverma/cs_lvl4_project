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

from sklearn.ensemble import GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier(n_estimators=5)
gbc_clf.fit(X_train, y_train)  
print('Gradient Boosting Results')
gbc_clf.score(X_test,y_test)

from sklearn.ensemble import AdaBoostClassifier
gbc_clf = AdaBoostClassifier(n_estimators=5,random_state=48)
gbc_clf.fit(X_train, y_train)
print('Ada Boosting Results')
gbc_clf.score(X_test,y_test)

# using random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc_clf = RandomForestClassifier(max_depth=1, random_state=48)
rfc_clf.fit(X_train, y_train)
print('Random Forest Results')
rfc_clf.score(X_test, y_test)

# using naive bayes
from sklearn.naive_bayes import GaussianNB
NB_clf = GaussianNB()
NB_clf.fit(X_train, y_train)
print('GaussianNB Results')
NB_clf.score(X_test, y_test)

# using NN Multi Layer Perceptron classifier
from sklearn.neural_network import MLPClassifier
NNMLP_clf = MLPClassifier(random_state=48, max_iter=50)
NNMLP_clf.fit(X_train, y_train)
print('NNMLP Classifier Results')
NNMLP_clf.score(X_test, y_test)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
xfit = clf.fit(X_train, y_train)
print('Support Vector Results')
clf.score(X_test,y_test)
