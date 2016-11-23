from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import yaml
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.optimizers import SGD

import cPickle as pickle
import scipy.sparse
import json

print("Loading data...")
#X, labels = load_data('train.csv', train=True)
#X, scaler = preprocess_data(X)
#y, encoder = preprocess_labels(labels)

X=[]
with open('train_newCosDesc_sparse_mat.dat', 'rb') as infile:
    X = pickle.load(infile)


X = X.astype('float32')
labels = pd.read_csv('labels.csv',index_col=False,header=None)
labels = np.array(labels).astype('int')

y = np.zeros((len(labels),38))
for i in xrange(0,len(labels)):
	y[i,labels[i]]=1


X_test=[]
with open('test_newCosDesc_sparse_mat.dat', 'rb') as infile:
    X_test = pickle.load(infile)

X = X.astype('float32')
#X_test, ids = load_data('test.csv', train=False)
#X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")


#############################################
model = Sequential()
model.add(Dense(200, input_dim=X.shape[1], init='glorot_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200, init='glorot_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(38, init='glorot_normal'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=8)

model.fit(X.toarray(), y, nb_epoch=500, batch_size=96, validation_split=0.2, shuffle=True,callbacks=[early_stopping])


# save as JSON
json_string = model.to_json()


with open('kerasModelJson.txt', 'w') as outfile:
    json.dump(json_string, outfile)


# save as YAML
yaml_string = model.to_yaml()

with open('kerasModel.yml', 'w') as outfile:
    outfile.write( yaml.dump(yaml_string, default_flow_style=True) )



score = model.predict_proba(X_test)
#score = model.evaluate(X_test.toarray(), y_test, batch_size=96)


preds = pd.DataFrame(score)
print "writing to xgbSolWithUpcLen.csv"
ss=pd.read_csv('sample_submission.csv')
preds.columns = ss.columns[1:]
preds['VisitNumber'] = ss['VisitNumber']
preds.set_index('VisitNumber', inplace=True)
preds.to_csv("kerasSol.csv")