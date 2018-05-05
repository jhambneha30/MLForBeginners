from gensim.models import Doc2Vec
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
import numpy
from  sklearn.model_selection import train_test_split
# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# random
import random
import pickle
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled

def countNumberOfLines (filename) :
	file  = open (filename , "r")
	data = file.readlines ()
	print len (data)
	return len (data)
	
if __name__ == "__main__":
	sources = {"stemmed_JOY_Phrases.txt":"Joy" , "stemmed_ANGER_Phrases.txt":"Anger" , "stemmed_SURPRISE_Phrases.txt" : "Surprise", "stemmed_FEAR_Phrases.txt" :"Fear" , "stemmed_SADNESS_Phrases.txt" : "Sadness" }
	# sources = {"Anger_10k.txt": "Anger" , "Fear_10k.txt":"Fear" , "Joy_10k.txt": "Joy" , "Sadness_10k.txt" : "Sadness" , "Surprise_10k.txt" : "Surprise" }
	sentences = LabeledLineSentence(sources)
	model = Doc2Vec(min_count=1, window=10, size=1000, sample=1e-4, negative=5, workers=7)
	model.build_vocab(sentences.to_array())
	for epoch in range(30):
		model.train(sentences.sentences_perm(), total_examples = model.corpus_count, epochs = model.iter)
	model.save("TE_1000.pkl")
	model = Doc2Vec.load("TE_1000.pkl")
	#print model.most_similar('anger')
	#print model.docvecs.doctags
	#print model.docvecs["Joy_87"]
	print model.corpus_count
	total = model.corpus_count
	train_arrays = numpy.zeros((total, 1000))
	train_labels = numpy.zeros(total)
	
	# dict = {"Anger" : countNumberOfLines("stemmed_ANGER_Phrases.txt") ,	"Fear" : countNumberOfLines ("stemmed_FEAR_Phrases.txt"),"Joy": countNumberOfLines("stemmed_JOY_Phrases.txt") , "Sadness" : countNumberOfLines ("stemmed_SADNESS_Phrases.txt"), "Surprise":countNumberOfLines("stemmed_SURPRISe_Phrases.txt") } 
	dict = {"Anger":1017 , "Fear":1986 , "Joy":1980 , "Sadness":1982 , "Surprise":1642 }
	k =0
	l = 0
	for i,m in dict.items():
		for j in range(m):
			train_arrays[l] = model.docvecs[i+"_" + str(j)]
			train_labels[l] = k
			l+=1
		k+=1
	#print set(train_labels)
	#print train_arrays
	X = train_arrays
	Y = train_labels
	zippedXY = zip(X,Y)
	pickle.dump(zippedXY,open("zippedXY_1000.p","wb"))
	
	#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	#classifier = LinearSVC( random_state = 4)
	#classifier = LogisticRegression()
	#classifier = svm.SVC(kernel = "linear")
	#classifier = svm.SVC()
	# classifier = LinearDiscriminantAnalysis()
	# classifier.fit(X_train, y_train)
	# print classifier.score(X_test, y_test)