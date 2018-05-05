from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def countvect(messages):
	vect = CountVectorizer(max_features=10)
	vect.fit(messages)
	feats = vect.get_feature_names()
	return feats


def tfidfvect(messages):
	vect = TfidfVectorizer(max_features=400)
	vect.fit(messages)
	feats = vect.get_feature_names()
	return feats

if __name__ == '__main__':
	files = ["stemmed_ANGER_Phrases.txt","stemmed_FEAR_Phrases.txt","stemmed_JOY_Phrases.txt",
				"stemmed_SADNESS_Phrases.txt","stemmed_SURPRISE_Phrases.txt"]
	## only for anger
	angerlines = open("stemmed_ANGER_Phrases.txt","rb").readlines()
	fearlines = open("stemmed_FEAR_Phrases.txt","rb").readlines()
	joylines = open("stemmed_JOY_Phrases.txt","rb").readlines()
	sadlines = open("stemmed_SADNESS_Phrases.txt","rb").readlines()
	surpriselines = open("stemmed_SURPRISE_Phrases.txt","rb").readlines()
	print "angerlines:",len(angerlines)
	print "fearlines:",len(fearlines)
	print "joylines:",len(joylines)
	print "sadlines:",len(sadlines)
	print "surpriselines:",len(surpriselines)
	#"""
	angertrain,feartrain,joytrain,sadtrain,surprisetrain = angerlines[:800],fearlines[:1600],joylines[:1600],sadlines[:1600],surpriselines[:1400]
	angertest,feartest,joytest,sadtest,surprisetest = angerlines[800:],fearlines[1600:],joylines[1600:],sadlines[1600:],surpriselines[1400:]
	test = angertest+feartest+joytest+sadtest+surprisetest
	angerfeatures = set(tfidfvect(angertrain))
	fearfeatures = set(tfidfvect(feartrain))
	joyfeatures = set(tfidfvect(joytrain))
	sadfeatures = set(tfidfvect(sadtrain))
	surprisefeatures = set(tfidfvect(surprisetrain))
	print "angerfeatures:",angerfeatures
	print "fearfeatures:",fearfeatures
	print "joyfeatures:",joyfeatures
	print "sadfeatures:",sadfeatures
	print "surprisefeatures:",surprisefeatures

	
	correct = 0
	print "len(test):",len(test)
	for t in test:
		## for anger
		print "sentence:",t
		words = set(t.split())
		countanger = len(words.intersection(angerfeatures))
		countfear = len(words.intersection(fearfeatures))
		countjoy = len(words.intersection(joyfeatures))
		countsad = len(words.intersection(sadfeatures))
		countsurprise = len(words.intersection(surprisefeatures))
		list_count = [countanger,countfear,countjoy,countsad,countsurprise]
		print "list_count:",list_count
		if (t in angertest) and (countanger == max(list_count)) and countanger>0:
			correct += 1
			continue
		elif (t in feartest) and (countfear == max(list_count)) and countfear > 0:
			correct += 1
			continue
		elif (t in joytest) and (countjoy == max(list_count)) and countjoy > 0:
			correct += 1
			continue
		elif (t in sadtest) and (countsad == max(list_count)) and countsad > 0:
			correct += 1
			continue
		elif (t in surprisetest) and (countsurprise == max(list_count)) and countsurprise > 0:
			correct += 1



	print "correct = ",correct
	print "Total accuracy = ",float(correct)/len(test)
	#"""