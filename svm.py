import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
import cPickle as pickle
import scipy.sparse
from sklearn.multiclass import OneVsRestClassifier

X=[]
with open('train_newFeat_sparse_mat.dat', 'rb') as infile:
	X = pickle.load(infile)

y = pd.read_csv('labels.csv',index_col=False,header=None)
y = np.array(y).astype('int')
y_temp = y[:,0]
y=y_temp

model = OneVsRestClassifier(SVC(C=100.0, gamma = 0.1, probability=True, verbose=1,kernel='linear'),n_jobs=-1)

model.fit(X, y)


X_test=[]
with open('test_newFeat_sparse_mat.dat', 'rb') as infile:
        X_test = pickle.load(infile)

from sklearn.externals import joblib
joblib.dump(model, 'svmModelUbuntu.pkl') 