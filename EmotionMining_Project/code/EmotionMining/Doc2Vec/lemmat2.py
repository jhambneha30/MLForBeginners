from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pickle
from nltk.corpus import wordnet as wn
from nltk.stem import *
from nltk.stem.porter import *
import msvcrt 

def countvect(messages):
	vect = CountVectorizer(max_features=50)
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

def getSynonyms (word):
	syns = wn.synsets( word )
	stemmer = PorterStemmer()
	synonyms = []
	for syn in syns:
		for l in syn.lemmas():
			synonyms.append (stemmer.stem(l.name()) )
	synonyms = list (set (synonyms))
	return synonyms	
	
def convertXY(sentenceLabelList,features):
	X,Y=[],[]
	#ctr = 0
	for sent in sentenceLabelList:
		featlist = []
		#ctr += 1
		#print sent[0]
		#print "Synonyms"
		for ff in features:
			#print getSynonyms(ff)
			if ff in sent[0]:
				featlist.append(1)
			elif len (commonFeatures( sent[0], getSynonyms (ff) ) ) >=1 :
				featlist.append(1)
			else:
				featlist.append(0)
		X.append(featlist)
		Y.append(sent[1])
		# if ctr == 10:
			# break
	#print X
		
	return X,Y
			


if __name__ == "__main__":

	files = ["ANGER_Phrases.txt10k.txt","FEAR_Phrases.txt10k.txt","JOY_Phrases.txt10k.txt",
				"SADNESS_Phrases.txt10k.txt","SURPRISE_Phrases.txt10k.txt"]

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
		lines = open(f,"rb").readlines()
		for line in lines:
			sentenceLabelList.append((line,label))
		features = tfidfvect(lines)
		tfidffeatures += features
	adjfeatures = open("stemmedFeatures.txt","rb").readlines()
	
	#print "tfidffeatures=",tfidffeatures
	#print "adjfeatures=",adjfeatures

	for ff in range(len(tfidffeatures)):
		tfidffeatures[ff] = str(tfidffeatures[ff])

	for ff in range(len(adjfeatures)):
		adjfeatures[ff] = adjfeatures[ff].replace("\n","")
		adjfeatures[ff] = adjfeatures[ff].strip()

	f1 = open("tfidf.txt","wb")
	f1.write( str (tfidffeatures) )
	f1.close()
	f1 = open("adjective.txt","wb")
	f1.write( str( adjfeatures ) )
	f1.close()
	#print "tfidffeatures=",tfidffeatures
	#msvcrt.raw_input()
	#print "adjfeatures=",adjfeatures

	finalfeatures = commonFeatures(tfidffeatures,adjfeatures)
	print "finalfeatures=",finalfeatures
	#print sentenceLabelList
	X,Y = convertXY(sentenceLabelList,finalfeatures)
	zippedXY = zip(X,Y)
	pickle.dump(zippedXY,open("zippedXY.p","wb"))