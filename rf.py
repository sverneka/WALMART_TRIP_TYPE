######################random forest ###########################
import numpy as np
import pandas as pd
import cPickle as pickle
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.externals import joblib

train=[]
with open('train_sparse_mat.dat', 'rb') as infile:
    train = pickle.load(infile)

train = train.tocsr()
num_features = train.shape[1]
num_classes=38
labels = pd.read_csv('labels.csv',index_col=False,header=None)

labels = np.array(labels).astype('int')

train_val = train[75000:,:]
labels_val = labels[75000:,0]

train = train[:75000,:]
labels = labels[:75000,0]


clf = RandomForestClassifier(n_estimators=2000, oob_score=True, verbose=1, n_jobs=-1, bootstrap=True)
clf.fit(train, labels)
#val_probs = clf.predict_proba(train_val)
print "accuracy is = ",accuracy
joblib.dump(clf, 'rfModel2000.pkl')

pred_label_val = clf.predict(train_val)

#sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
#sig_clf.fit(X_valid, y_valid)
#sig_clf_probs = sig_clf.predict_proba(X_test)
#sig_score = log_loss(y_test, sig_clf_probs)

#val_score = log_loss(labels_val, val_probs)

#print val_score

accuracy = len(np.where(pred_label_val == labels_val)[0])*1.0/len(labels_val)

print "accuracy is = ",accuracy
joblib.dump(clf, 'rfModel2000.pkl') 

#clf = RandomForestClassifier(n_estimators=5000, oob_score=True, verbose=1, n_jobs=-1, bootstrap=True)
