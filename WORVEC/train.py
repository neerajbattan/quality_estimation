from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
from scipy.stats.stats import pearsonr
import pickle 
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
import pickle
import sys

train_features = np.load(sys.argv[1])
test_features = np.load(sys.argv[4])

labels = open(sys.argv[3]).read().split('\n')
#labels = labels.tolist()
print labels
labels = [float(i) for i in labels[:-1]]

test_labels = open(sys.argv[6]).read().split('\n')
test_labels = [float(i) for i in test_labels[:-1]]

def makeFeatures(file):
    otherFeatures = open(file).read()
    otherFeatures = otherFeatures.split('\n')
    tmp = [] 
    for i in otherFeatures:
        if len(i)==0:
            continue
        vec = np.array(i.split(),dtype='float64')
        tmp.append(vec)
    return np.array(tmp)

train_features = np.concatenate([makeFeatures(sys.argv[2]),train_features],axis=1)
test_features = np.concatenate([makeFeatures(sys.argv[5]),test_features],axis=1)
#clf = pickle.load(open(sys.argv[7]))

clf = RandomForestRegressor(n_jobs=5)
clf.fit(train_features, labels)
pickle.dump(clf,open(sys.argv[6],'w'))

y = clf.predict(test_features)
#for i in range(len(y)):
#    print(y[i], labels[24000+i])

print pearsonr(y,test_labels)

