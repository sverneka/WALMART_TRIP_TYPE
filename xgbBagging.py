import numpy as np
import pandas as pd
from scipy.special import expit
import random
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, MDS
from sklearn.decomposition import TruncatedSVD
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.cross_validation import train_test_split
import cPickle as pickle
import scipy.sparse
import array
from sknn.mlp import Classifier, Layer
from scipy.sparse import hstack
import xgboost as xgb

path='/home/lucifier/walmart'

random.seed(21)
np.random.seed(21)

def load_train_data(path,modelNo=1):
    X=[]
    with open(path+'/train_newFeat_sparse_mat.dat', 'rb') as infile:
        X = pickle.load(infile)
    random.seed(modelNo)
    np.random.seed(modelNo)
    r = random.sample(xrange(0,X.shape[1]),int(round(0.8*X.shape[1])))
    X = X[:,r]
    y = pd.read_csv(path+'/labels.csv',index_col=False,header=None)
    y = np.array(y).astype('int')
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=modelNo,stratify=y)
    xgtrain = xgb.DMatrix(X_train,y_train)
    xgval = xgb.DMatrix(X_val,y_val)
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    params = {"objective": "multi:softprob",
              "eta": 0.005,# used to be 0.2 or 0.1
              "max_depth": 5, # used to be 5 or 6
              "min_child_weight": 1,
                      "max_delta_step": 6,
              "silent": 1,
              "colsample_bytree": 0.7,
                      "subsample": 0.8,
                      "eval_metric" : "mlogloss",
              "seed": 1,
                      "num_class": 38}
    plst = list(params.items())
    num_rounds = 11000

    rbm1 = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    #rbm1 = SVC(C=100.0, gamma = 0.1, probability=True, verbose=1).fit(X[0:9999,:], y[0:9999])
    #rbm2 = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features='auto', bootstrap=False, oob_score=False, n_jobs=1, verbose=1).fit(X[0:9999,:], y[0:9999])
    #rbm3 = GradientBoostingClassifier(n_estimators=50,max_depth=11,subsample=0.8,min_samples_leaf=5,verbose=1).fit(X[0:9999,:], y[0:9999])
    with open(path+'/test_newFeat_sparse_mat.dat', 'rb') as infile:
        Y = pickle.load(infile)
    Y = Y[:,r]
    xgtest=xgb.DMatrix(Y)
    preds1=rbm1.predict(xgtest)

    p1 = pd.DataFrame(preds1)
    p1.to_csv("p1_"+str(modelNo)+".csv",index=None,header=None)
    return preds1

path='/home/lucifier/walmart'
num_runs = 20
test=[]
with open(path+'/test_newFeat_sparse_mat.dat', 'rb') as infile:
	test = pickle.load(infile)


y_prob = np.zeros((test.shape[0],38))
for jj in xrange(num_runs):
  print(jj)
  preds = load_train_data(path,jj+1)
  y_prob = y_prob + preds
  preds = pd.DataFrame(y_prob/(jj+1.0))
  ss=pd.read_csv(path+'/sample_submission.csv')
  preds.columns = ss.columns[1:]
  preds['VisitNumber'] = ss['VisitNumber']
  preds.set_index('VisitNumber', inplace=True)
  preds.to_csv('enSol'+str(jj+1)+'.csv')


y_prob = y_prob/(num_runs+1.0)

preds = pd.DataFrame(y_prob)
print "writing to xgbSolWithUpcLen.csv"
ss=pd.read_csv(path+'/sample_submission.csv')
preds.columns = ss.columns[1:]
preds['VisitNumber'] = ss['VisitNumber']
preds.set_index('VisitNumber', inplace=True)
preds.to_csv('xgbBaggingResults.csv')