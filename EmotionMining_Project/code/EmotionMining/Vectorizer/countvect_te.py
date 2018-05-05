from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import h5py
import pickle


def countvect(messages):
	vect = CountVectorizer(max_features=500)
	vect.fit(messages)
	feats = vect.get_feature_names()
	return feats


def tfidfvect(messages):
	vect = TfidfVectorizer(max_features=None)
	vect.fit(messages)
	feats = vect.get_feature_names()
	return feats


def commonFeatures(list1,list2):
	set1,set2 = set(list1),set(list2)
	commonf = set1.intersection(set2)
	return list(commonf)

def featureIndexMapping(features):
	feat2index,index2feat = {},{}
	for ff in range(len(features)):
		feat2index[features[ff]] = ff
		index2feat[ff] = features[ff]
	return feat2index,index2feat

def convertXY(sentenceLabelList,features):
	X,Y=[],[]
	for sent in sentenceLabelList:
		featlist = []
		for ff in features:
			if ff in sent[0]:
				featlist.append(1)
			else:
				featlist.append(0)
		X.append(featlist)
		Y.append(sent[1])
	#print X
	return X,Y
			


if __name__ == "__main__":

	files = ["stemmed_ANGER_Phrases.txt","stemmed_FEAR_Phrases.txt","stemmed_JOY_Phrases.txt",
				"stemmed_SADNESS_Phrases.txt","stemmed_SURPRISE_Phrases.txt"]

	sentenceLabelList = []

	tfidffeatures,adjfeatures = [],[]
	current_label = ""
	for f in files:
		if "ANGER" in f:
			label = "ANGER"
		if "FEAR" in f:
			label = "FEAR"
		if "JOY" in f:
			label = "JOY"
		if "SADNESS" in f:
			label = "SADNESS"
		if "SURPRISE" in f:
			label = "SURPRISE"
		lines = open("../Preprocess/te/"+f,"rb").readlines()
		for line in lines:
			sentenceLabelList.append((line,label))
		features = countvect(lines)
		tfidffeatures += features
	adjfeatures = open("../Preprocess/te/stemmedFeatures.txt","rb").readlines()
	
	#print "tfidffeatures=",tfidffeatures
	#print "adjfeatures=",adjfeatures

	for ff in range(len(tfidffeatures)):
		tfidffeatures[ff] = str(tfidffeatures[ff])

	for ff in range(len(adjfeatures)):
		adjfeatures[ff] = adjfeatures[ff].replace("\n","")

	tfidffeatures = set(tfidffeatures)
	print "tfidffeatures=",tfidffeatures
	print "length tfidffeatures=",len(tfidffeatures)
	#print "adjfeatures=",adjfeatures

	#finalfeatures = commonFeatures(tfidffeatures,adjfeatures)
	pickle.dump(tfidffeatures,open("finalfeatures_te.p","wb"))
	#print "finalfeatures=",finalfeatures
	#print sentenceLabelList
	X,Y = convertXY(sentenceLabelList,tfidffeatures)

	## saving to pickl file
	#zippedXY = zip(X,Y)
	#pickle.dump(zippedXY,open("zippedXY_te.p","wb"))

	## saving to h5py file
	h5f = h5py.File('XY_te.h5', 'w')
	h5f.create_dataset('X', data=X)
	h5f.create_dataset('Y', data=Y)
	h5f.close()
