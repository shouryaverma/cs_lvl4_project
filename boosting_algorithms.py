from sklearn.model_selection import LeaveOneOut
import numpy as np
import glob

values = np.empty(shape=[0, 189])
sample_val_all = glob.glob('/nfs/sampled_val_all.csv')

for j in sample_val_all:
    print('Loading ', j)
    csvrows = np.loadtxt(j, delimiter=',')
    values = np.append(values, csvrows, axis=0)
    
print(values.shape)
    
X = values[:,:-2]
y = values[:,-2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=48)

from sklearn.ensemble import GradientBoostingClassifier
gbc_clf = GradientBoostingClassifier(n_estimators=5)
gbc_clf.fit(X_train, y_train)  
print('Gradient Boosting Results')
gbc_clf.score(X_test,y_test)

from sklearn.ensemble import AdaBoostClassifier
gbc_clf = AdaBoostClassifier(n_estimators=5,random_state=48
gbc_clf.fit(X_train, y_train)
print('Ada Boosting Results')
gbc_clf.score(X_test,y_test)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
clf.score(X_test,y_test)
