from flask import Flask, render_template , request
from nltk.stem import *
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
import pickle
from sklearn.externals import joblib
 
app = Flask(__name__)

def preprocess_unigram(text):
	stop_words = set(stopwords.words('english'))
	stemmer = PorterStemmer()
	for c in string.punctuation:
		text = text.replace(c, "")
	text = text.lower()
	word_tokens = word_tokenize(text)
	stemmed_sent = [stemmer.stem(word) for word in word_tokens]
	filtered_sentence = [w for w in stemmed_sent if not w in stop_words]
	filtered_sentence = " ".join(filtered_sentence)
	return filtered_sentence

def preprocess_bigram(sentences):
	importantwords = {}
	importantwords["JJ"],importantwords["RB"],importantwords["VB"] = [],[],[]
	stop_words_list = stopwords.words('english')
	stop_words_list.remove('not')
	stop_words_list.remove('didn')
	stop_words_list.remove('hasn')
	stop_words_list.remove('mustn')
	stop_words_list.remove('isn')
	stop_words_list.remove('mightn')
	stop_words_list.remove('wasn')
	stop_words_list.remove('aren')
	stop_words_list.remove('wouldn')
	stop_words_list.remove('weren')
	stop_words_list.remove('doesn')
	stop_words_list.remove('shan')
	stop_words_list.remove('needn')
	stop_words_list.remove('haven')
	stop_words = set(stop_words_list)
	# print stop_words
	stemmed = []
	stemmer = PorterStemmer()
	ii = 0
	for sent in sentences:
		print "Sentence = ",(ii+1)
		word_tokens = word_tokenize(sent)
		postags =  nltk.pos_tag(word_tokens)
		for pt in postags:
			#print pt[0]
			try:
				importantwords[str(pt[1])] += [str(pt[0])]
			except KeyError:
				pass
		stemmed_sent = list()
		for word in word_tokens:
			try:
				word = stemmer.stem(word)
			except UnicodeDecodeError:
				print "UnicodeDecodeError @", ii
				pass
			stemmed_sent.append(word)    
		# stemmed_sent = [stemmer.stem(word) for word in word_tokens]
		filtered_sentence = [w for w in stemmed_sent if not w in stop_words]
		try:
			filtered_sentence = " ".join(filtered_sentence)
		except UnicodeDecodeError:
			pass
		stemmed.append(filtered_sentence)
		ii+=1

	# Removing punctuation and converting to lower case
	processed = list()
	for line in stemmed:
		for c in string.punctuation:
			line = line.replace(c, "")
			line = line.lower()
		processed.append(line)
	return processed

def get_bigram_vec(text):
	tokens = nltk.word_tokenize(text)
	bigrams = list(nltk.bigrams(tokens))
	features = pickle.load(open("../Vectorizer/bigram_finalfeatures.p","rb"))
	X = []
	for ff in features:
		if ff in bigrams:
			X.append(1)
		else:
			X.append(0)
	return X

def get_unigram_vec(filtered_sentence):
	features = pickle.load(open("../Vectorizer/unigram_finalfeatures_prev.p","rb"))
	#print "features:",features
	X = []
	for ff in features:
		if ff in filtered_sentence:
			X.append(1)
		else:
			X.append(0)
	return X

@app.route("/")
def main():
	_text = ""
	return render_template('home.html',text=_text)

@app.route('/emotion',methods=['POST'])
def GetEmotion():
	_text = request.form['text']
	print("TEXT first = ",_text)
	notwords = ['not','didn','hasn','mustn','isn','mightn','wasn','aren','wouldn','weren','doesn','shan','needn','haven']
	preprocessed_text_unigram = preprocess_unigram(_text)
	print "preprocessed_text_unigram:",preprocessed_text_unigram
	X_text_uni = get_unigram_vec(preprocessed_text_unigram)

	#preprocessed_text_bigram = preprocess_bigram([_text])
	#X_text_bi = get_bigram_vec(preprocessed_text_bigram[0])

	#X_text = X_text_uni + X_text_bi
	print "X_uni : ",len(X_text_uni)
	#print "X_bi : ",len(X_text_bi)
	#print "X_vector : ",len(X_text)
	print "X_text_uni.count(1) : ",X_text_uni.count(1)
	if X_text_uni.count(1) == 0:
		return render_template('home.html',text=_text,emotion="Neutral")
	
	else:
		svm_model = pickle.load(open("../SVM/svm_model.p","rb"))
		#svm_model = joblib.load("../Best_Model/nn_wff10_unibi.model","r")
		predictedY = svm_model.predict([X_text_uni])
		setofinput = set(_text.split(" "))
		setofno = set(notwords)
		print "setofinput:",setofinput
		print "setofno:",setofno
		if len(setofno.intersection(setofinput)) > 0:
			print "There is a negative word in the sentence."
			if predictedY[0] == "ANGER":
				predictedY[0] = "NEUTRAL"
			elif predictedY[0] == "JOY":
				predictedY[0] = "SADNESS"
			elif predictedY[0] == "SADNESS":
				predictedY[0] = "JOY"
			elif predictedY[0] == "FEAR":
				predictedY[0] = "NEUTRAL"
			elif predictedY[0] == "SURPRISE":
				predictedY[0] = "NEUTRAL"

		if X_text_uni.count(1) <= 3:
			return render_template('home.html',text=_text,emotion=str(predictedY[0]+" (Less input data to correctly predict)"))
		else:
			return render_template('home.html',text=_text,emotion=predictedY[0])

if __name__ == "__main__":
	app.run()
