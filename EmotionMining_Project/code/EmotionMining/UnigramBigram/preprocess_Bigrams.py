import sys
from nltk.stem import *
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import pickle
import random

# Required for Neha's laptop
sys.path.append("/usr/local/lib/python2.7/dist-packages")


def extractN(filename,n=None):
    lines = open("../Dataset/we_feel_fine/Phrases/"+filename,"rb").readlines()
    random.shuffle(lines)
    if n==None:
        return lines
    return lines[:n]

def stemmingAndStopwords(sentences):
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
    #print "importantwords=",importantwords
    return stemmed,importantwords

def removePunctAndLower(sentences):
    # Removing punctuation and converting to lower case
    processed = list()
    for line in sentences:
        for c in string.punctuation:
            line = line.replace(c, "")
            line = line.lower()
        processed.append(line)
    return processed

def printDictDetails(mydict):
    print "____Key___ : ___Number of Values___"
    for key in mydict.keys():
        print key," : ",len(mydict[key])

def saveValuesDictStemmed(mydict):
    stemmer = PorterStemmer()
    vfile = open("stemmedFeatures.txt","wb")
    vfile = open("stemmedFeatures.txt","a")
    for key in mydict.keys():
        values = mydict[key]
        for v in values:
            stemmedv = stemmer.stem(v)
            vfile.write(stemmedv+"\n")
    vfile.close()

if __name__ == "__main__":
    
    files = ["ANGER_Phrases.txt","FEAR_Phrases.txt","JOY_Phrases.txt","SADNESS_Phrases.txt","SURPRISE_Phrases.txt"]

    importantFeatures = {}
    importantFeatures["JJ"],importantFeatures["RB"],importantFeatures["VB"] = [],[],[]

    for f in files:

        # Extract only specified number of sentences
        sentences = extractN(f, 10000)

        # Removing punctuation and converting to lower case
        processed = removePunctAndLower(sentences)

        # Stemming and Removing stopwords, Return important words
        stemmed_sentence,importantwords = stemmingAndStopwords(processed)

        # Removing duplicates from JJ,RB,VB(importantwords) dictionary
        for key in importantwords.keys():
            list_exist = importantwords.get(key)
            list_duplicate_remove = list(set(list_exist))
            importantFeatures[key] += list_duplicate_remove
        #print "importantwords=",importantFeatures
        
        # print "without removing duplicates: ", len(stemmed_sentence)
        # Removing duplicates
        stemmed_unique = list(set(stemmed_sentence))
        #print "after removing duplicates: ", stemmed_unique
        print("Writing to file....  ", f)
        of = open("bi10000_stemmed_"+f, "wb")
        for sent in stemmed_unique:
            of.write(sent+"\n")
        of.close()
    print "importantFeatures:",importantFeatures
    pickle.dump(importantFeatures,open("importantFeatures.p","wb"))
    printDictDetails(importantFeatures)
    saveValuesDictStemmed(importantFeatures)

        

