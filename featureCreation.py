import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import copy
import array

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_cols = train.columns
test_cols = test.columns

tr = np.array(train)
te = np.array(test)

weekDays = ('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday')

i=0
for day in weekDays:
	ind = np.where(tr[:,2] == day)[0]
	tr[ind,2] = i
	ind = np.where(te[:,1] == day)[0]
	te[ind,1] = i
	i = i+1

col = []
for i in xrange(0,len(tr[:,-2])):
	if not isinstance(tr[i,-2],basestring):
		col.append(i)

tr[col,-2] = 'UNKNOWN'

col = []
for i in xrange(0,len(te[:,-2])):
	if not isinstance(te[i,-2],basestring):
		col.append(i)

te[col,-2] = 'UNKNOWN'



depDesc = np.unique(np.concatenate((tr[:,-2], te[:,-2]), axis=0))#this is already sorted 69 unique values including UNKNOWN
descDict = {}
count = 0
for desc in depDesc:
	descDict[desc] = count
	ind = np.where(tr[:,-2] == desc)[0]
	tr[ind,-2] = count
	ind = np.where(te[:,-2] == desc)[0]
	te[ind,-2] = count
	count = count+1


#process UPC
utr = tr[:,-4].astype('float')
ute = te[:,-4].astype('float')

ind = np.where(np.isnan(utr) == True)[0]
tr[ind,-4] = -999
tr[:,-4] = tr[:,-4].astype('int')
ind = np.where(np.isnan(ute) == True)[0]
te[ind,-4] = -999
te[:,-4] = te[:,-4].astype('int')



#process finelineNumber
ftr = tr[:,-1].astype('float')
fte = te[:,-1].astype('float')

ind = np.where(np.isnan(ftr) == True)[0]
tr[ind,-1] = -999
tr[:,-1] = tr[:,-1].astype('int')
ind = np.where(np.isnan(fte) == True)[0]
te[ind,-1] = -999
te[:,-1] = te[:,-1].astype('int')

fineNum = np.unique(np.concatenate((tr[:,-1], te[:,-1]), axis=0))#5354 unique values including -999
trFineNum = np.copy(tr[:,-1])
teFineNum = np.copy(te[:,-1])
count = 0
fineNumDict={}

for fN in fineNum:
	fineNumDict[fN] = count
	count = count+1

for i in xrange(0,tr.shape[0]):
	tr[i,-1] = fineNumDict[tr[i,-1]]

for i in xrange(0,te.shape[0]):
	te[i,-1] = fineNumDict[te[i,-1]]

visitNumTr = np.unique(tr[:,1])
visitNumTe = np.unique(te[:,0])

descMatTr = np.zeros((len(visitNumTr), len(depDesc)))
descMatTe = np.zeros((len(visitNumTe), len(depDesc)))

descMatPosTr = np.zeros((len(visitNumTr), len(depDesc)))
descMatPosTe = np.zeros((len(visitNumTe), len(depDesc)))

descMatNegTr = np.zeros((len(visitNumTr), len(depDesc)))
descMatNegTe = np.zeros((len(visitNumTe), len(depDesc)))

fineMatTr = np.zeros((len(visitNumTr), len(fineNum)))
fineMatTe = np.zeros((len(visitNumTe), len(fineNum)))

wDayTr = np.zeros((len(visitNumTr),1))
wDayTe = np.zeros((len(visitNumTe),1))

vNumTr = np.zeros((len(visitNumTr),1))
vNumTe = np.zeros((len(visitNumTe),1))

tripType = np.zeros((len(visitNumTr)))

count = 0
for i in xrange(0,tr.shape[0]):
	cVN = tr[i,1]
	if(cVN == visitNumTr[count]):
		vNumTr[count] = cVN
		tripType[count] = tr[i,0]
		wDayTr[count] = tr[i,2]
		descMatTr[count,tr[i,-2]] += tr[i,-3]
		if(tr[i,3] < 0):
			descMatNegTr[count,tr[i,-2]] += tr[i,-3]
		else:
			descMatPosTr[count,tr[i,-2]] += tr[i,-3]
		fineMatTr[count,tr[i,-1]] += tr[i,-3]
	else:
		count = count +1
		vNumTr[count] = cVN
		tripType[count] = tr[i,0]
		wDayTr[count] = tr[i,2]
		descMatTr[count,tr[i,-2]] += tr[i,-3]
		if(tr[i,3] < 0):
			descMatNegTr[count,tr[i,-2]] += tr[i,-3]
		else:
			descMatPosTr[count,tr[i,-2]] += tr[i,-3]
		fineMatTr[count,tr[i,-1]] += tr[i,-3]

count = 0
for i in xrange(0,te.shape[0]):
	cVN = te[i,0]
	if(cVN == visitNumTe[count]):
		vNumTe[count] = cVN
		wDayTe[count] = te[i,1]
		descMatTe[count,te[i,-2]] += te[i,-3]
		if(te[i,3] < 0):
			descMatNegTe[count,te[i,-2]] += te[i,-3]
		else:
			descMatPosTe[count,te[i,-2]] += te[i,-3]
		fineMatTe[count,te[i,-1]] += te[i,-3]
	else:
		count = count +1
		vNumTe[count] = cVN
		wDayTe[count] = te[i,1]
		descMatTe[count,te[i,-2]] += te[i,-3]
		if(te[i,3] < 0):
			descMatNegTe[count,te[i,-2]] += te[i,-3]
		else:
			descMatPosTe[count,te[i,-2]] += te[i,-3]
		fineMatTe[count,te[i,-1]] += te[i,-3]

tripTypeDict = {}
count = 0
for i in np.unique(tripType):
	tripTypeDict[i] = count
	count = count+1

for i in xrange(0,len(tripType)):
	tripType[i] = tripTypeDict[tripType[i]]


itemsBotTr = np.zeros((len(visitNumTr),1))
itemsBotTe = np.zeros((len(visitNumTe),1))

itemsRetTr = np.zeros((len(visitNumTr),1))
itemsRetTe = np.zeros((len(visitNumTe),1))

itemsTr = np.zeros((len(visitNumTr),1))
itemsTe = np.zeros((len(visitNumTe),1))

for i in xrange(0,descMatTr.shape[0]):
	ind = np.where(descMatTr[i,:] < 0)
	itemsRetTr[i] = np.sum(descMatTr[i,ind])
	ind = np.where(descMatTr[i,:] > 0)
	itemsBotTr[i] = np.sum(descMatTr[i,ind])
	itemsTr[i] = np.sum(descMatTr[i,:])



for i in xrange(0,descMatTe.shape[0]):
	ind = np.where(descMatTe[i,:] < 0)
	itemsRetTe[i] = np.sum(descMatTe[i,ind])
	ind = np.where(descMatTe[i,:] > 0)
	itemsBotTe[i] = np.sum(descMatTe[i,ind])
	itemsTe[i] = np.sum(descMatTe[i,:])

######correlation features for UPC##################
#as of now ignoring total number of elements.
def getMatch(uA, sA, uB, sB):
	matFract = 2.0*len(np.intersect1d(uA,uB))/(len(uA) + len(uB))
	return matFract

def getMatchDF(A, B):
	matFract = 1 - (1.0*np.sum(np.abs(A-B))/(np.sum(np.abs(A)) + np.sum(np.abs(B))))
	return matFract



gpFTr = []
gpFTe = []
upcTr = []
scTr = []
ddTr = []
fnTr = []
upcTe = []
scTe = []
ddTe = []
fnTe = []
upcTemp = []
scTemp = []
ddTemp = []
fnTemp = []

count = 0
cVN = 0
for i in xrange(0,tr.shape[0]):
	cVN = tr[i,1]
	if(cVN == visitNumTr[count]):
		upcTemp.append(tr[i,-4])
		scTemp.append(tr[i,-3])
		ddTemp.append(tr[i,-2])
		fnTemp.append(tr[i,-1])
	else:
		count = count + 1
		upcTr.append(upcTemp)
		scTr.append(scTemp)
		ddTr.append(ddTemp)
		fnTr.append(fnTemp)
		upcTemp = []
		scTemp = []
		ddTemp = []
		fnTemp = []
		upcTemp.append(tr[i,-4])
		scTemp.append(tr[i,-3])
		ddTemp.append(tr[i,-2])
		fnTemp.append(tr[i,-1])


upcTr.append(upcTemp)
scTr.append(scTemp)
ddTr.append(ddTemp)
fnTr.append(fnTemp)

for i in xrange(0,len(upcTr)):
	uniqueUpc = np.unique(upcTr[i])
	upc = np.array(upcTr[i])
	scTrTemp = []
	for u in uniqueUpc:
		ind = np.where(upc == u)[0]
		scTemp = np.array(scTr[i])[ind]
		if len(ind) > 1:
			scTrTemp.append(sum(scTemp))
		else:
			scTrTemp.append(scTemp)
	scTr[i] = scTrTemp
	upcTr[i] = uniqueUpc

#need to do the same for test data

classIndTr = []
classIndTe = []

for i in np.unique(tripType):
	ind = np.where(tripType[:] == i)[0]
	classIndTr.append(ind)


#get correlation matrix for UPC
upcCorrTr = []
upcCorrTe = []

tripTypeUnique = np.unique(tripType)
for i in xrange(0, visitNumTr.shape[0]):
	maxMatch = 0
	avgMatch = 0
	upcCorr = []
	for j in xrange(0, tripTypeUnique.shape[0]):
		maxMatch = 0
		avgMatch = 0
		for k in classIndTr[j]:
			if(k != i):
				matFract = getMatch(upcTr[i], scTr[i], upcTr[k], scTr[k])
				if(maxMatch < matFract):
					maxMatch = matFract
				avgMatch += matFract
		avgMatch = avgMatch*1.0/(len(classIndTr[j])-1)
		upcCorr.append(maxMatch)
		upcCorr.append(avgMatch)
	upcCorrTr.append(upcCorr)
	print i


descCorrTr = []
descCorrTe = []
for i in xrange(0, visitNumTr.shape[0]):
	descCorr = []
	for j in xrange(0, tripTypeUnique.shape[0]):
		maxMatch = 0
		avgMatch = 0
		for k in classIndTr[j]:
			if(k != i):
				matFract = getMatchDF(descMatTr[i,:], descMatTr[k,:])
				if(maxMatch < matFract):
					maxMatch = matFract
				avgMatch += matFract
		avgMatch = avgMatch*1.0/(len(classIndTr[j])-1)
		descCorr.append(maxMatch)
		descCorr.append(avgMatch)
	descCorrTr.append(descCorr)
	print i


fCorrTr = []
fCorrTe = []
for i in xrange(0, visitNumTr.shape[0]):
	maxMatch = 0
	avgMatch = 0
	fCorr = []
	for j in xrange(0, tripTypeUnique.shape[0]):
		maxMatch = 0
		avgMatch = 0
		for k in classIndTr[j]:
			if(k != i):
				matFract = getMatchDF(fineMatTr[i,:], fineMatTr[k,:])
				if(maxMatch < matFract):
					maxMatch = matFract
				avgMatch += matFract
		avgMatch = avgMatch*1.0/(len(classIndTr[j])-1)
		fCorr.append(maxMatch)
		fCorr.append(avgMatch)
	fCorrTr.append(fCorr)
	print i



finalMatTr = np.concatenate((vNumTr, wDayTr, itemsBotTr, itemsRetTr, itemsTr, descMatTr, descMatPosTr, descMatNegTr, fineMatTr, tripType),axis=1)

del(visitNum,visitNumTr, vNumTr, wDayTr, itemsBotTr, itemsRetTr, descMatTr, descMatPosTr, descMatNegTr, fineMatTr)

finalMatTr = pd.DataFrame(finalMatTr)
print "writing to trainMat.csv"
finalMatTr.to_csv('trainMat.csv',header=None, index=False)
del(finalMatTr)

finalMatTe = np.concatenate((vNumTe, wDayTe, itemsBotTe, itemsRetTe, itemsTe, descMatTe, descMatPosTe, descMatNegTe, fineMatTe),axis=1)

del(visitNum, visitNumTe, vNumTe, wDayTe, itemsBotTe, itemsRetTe, descMatTe, descMatPosTe, descMatNegTe, fineMatTe)

finalMatTe = pd.DataFrame(finalMatTe)
print "writing to testMat.csv"
finalMatTe.to_csv('testMat.csv',header=None, index=False)
del(finalMatTe)






















#get correlation matrix for UPC
upcCorrTr = []
upcCorrTe = []
tripType = tripType.astype('int')
tripTypeUnique = np.unique(tripType)
m = len(tripTypeUnique)
p = range(0, visitNumTr.shape[0])
for i in xrange(0, visitNumTr.shape[0]):
	upcCorr = []
	maxCorr = [0] * 10
	avgCorr = [0] * 10
	maxCorr = np.zeros((1,len(tripTypeUnique)))[0]
	avgCorr = np.zeros((1,len(tripTypeUnique)))[0]
	q = copy.deepcopy(p)
	del q[i]
	for j in q:
		matFract = 2.0*len(np.intersect1d(upcTr[i],upcTr[j]))/(len(upcTr[i]) + len(upcTr[j]))
		#matFract = getMatch(upcTr[i], scTr[i], upcTr[j], scTr[j])
		if(maxCorr[tripType[i][0]] < matFract):
			maxCorr[tripType[i][0]] = matFract
		avgCorr[tripType[i][0]] += matFract
	for k in tripTypeUnique:
		avgCorr[tripType[k][0]] = avgCorr[tripType[k][0]]*1.0/(len(classIndTr[tripType[k][0]])-1)
	upcCorrTr.append(np.concatenate((maxCorr, avgCorr),axis=1))
	print i








#get correlation matrix for UPC
upcCorrTr = []
upcCorrTe = []
tripType = tripType.astype('int')
tripTypeUnique = np.unique(tripType)
m = len(tripTypeUnique)
p = range(0, visitNumTr.shape[0])
tripType = array.array('i',tripType.tolist())
lenClassIndTr = [0] * m
lenClassIndTr = array.array('i',lenClassIndTr)
for i in xrange(0,m):
	lenClassIndTr[i] = len(classIndTr[i])

lenUpcTr = [0] * visitNumTr.shape[0]
lenUpcTr = array
for i in xrange(0, visitNumTr.shape[0]):
	lenUpcTr[i] = len(upcTr[i])


for i in xrange(0, visitNumTr.shape[0]):
	upcCorr = []
	maxCorr = [0] * m
	avgCorr = [0] * m
	q = range(0, visitNumTr.shape[0])
	del q[i]
	for j in q:
		matFract = 2.0*len(upcTr[i]&upcTr[j])/(lenUpcTr[i] + lenUpcTr[j])
		#matFract = getMatch(upcTr[i], scTr[i], upcTr[j], scTr[j])
		if(maxCorr[tripType[i]] < matFract):
			maxCorr[tripType[i]] = matFract
		avgCorr[tripType[i]] += matFract
	for k in tripTypeUnique:
		avgCorr[tripType[k]] = avgCorr[tripType[k]]*1.0/(lenClassIndTr[tripType[k]]-1)
	upcCorrTr.append(maxCorr.append(avgCorr))
	print i







