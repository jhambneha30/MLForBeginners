from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
import pickle

def countvect(messages):
    vect = CountVectorizer(max_features=50)
    vect.fit(messages)
    feats = vect.get_feature_names()
    return feats

def get_sentence_bigrams(messages):
    sentence_bigrams = []
    sentence_wise_bigrams = []
    for single in messages:
        tokens = nltk.word_tokenize(single)
        bigram = list(nltk.bigrams(tokens))
        # print bigram
        sentence_wise_bigrams.append(bigram)
        for b in bigram:
            sentence_bigrams.append(b)

    return sentence_bigrams, sentence_wise_bigrams

# def tfidfvect(messages):
#     # vect = TfidfVectorizer(max_features=None)
#     # vect.fit(messages)
#     bigrams_vector = []
#     for single in messages:
#         bigram_list = []
#         tokens = nltk.word_tokenize(single)
#         bigram = list(nltk.bigrams(tokens))
#         # print bigram
#         for a, b in bigram:
#             bigrams_vector.append((a, b))
#
#     features_vec = set(bigrams_vector)
#     # print "bigrams_vector complete: ", features_vec[:20]
#     # feats = vect.get_feature_names()
#     return features_vec


def commonFeatures(list1, list2_adj):
    commonf = []
    for a, b in list1:
        if a in list2_adj or b in list2_adj:
            commonf.append((a, b))
    print "commonf", commonf[: 20]
    return commonf

def keep_frequent_bigrams(bigram_list):
    feature_count = []
    for bi in set(bigram_list):
        feature_count.append((bi, bigram_list.count(bi)))

    feature_count.sort(key=lambda x: -x[1])
    print feature_count[:20]

    freq_feature_list = []
    for f in feature_count:
        if f[1] > 15:
            freq_feature_list.append(f[0])
    print freq_feature_list[:30]
    return freq_feature_list


def featureIndexMapping(features):
    feat2index,index2feat = {},{}
    for ff in range(len(features)):
        feat2index[features[ff]] = ff
        index2feat[ff] = features[ff]
    return feat2index,index2feat

def convertXY(sentenceWiseBigramList, features):
    # print "sent Bigram: ", sentenceWiseBigramList[:2]
    print "features: ", features[:40]
    X,Y=[], []
    for sent in sentenceWiseBigramList:
        # print "sentence bigrams: ", sent
        featlist = []
        for tuple in features:
            if tuple in sent[0]:
                featlist.append(1)
            else:
                featlist.append(0)

        # print "count of 1s: ", featlist.count(1)
        X.append(featlist)
        Y.append(sent[1])
    print X[:2]
    print Y[:2]
    return X, Y



if __name__ == "__main__":

    # files = ["stemmed_ANGER_Phrases.txt","stemmed_FEAR_Phrases.txt","stemmed_JOY_Phrases.txt",
    #             "stemmed_SADNESS_Phrases.txt","stemmed_SURPRISE_Phrases.txt"]

    files = ["bi10000_stemmed_ANGER_Phrases.txt", "bi10000_stemmed_FEAR_Phrases.txt", "bi10000_stemmed_JOY_Phrases.txt",
            "bi10000_stemmed_SADNESS_Phrases.txt", "bi10000_stemmed_SURPRISE_Phrases.txt"]

    # sentenceLabelList = []
    sentence_bigrams_complete = []
    sentence_wise_bigrams_complete = []
    tfidffeatures, adjfeatures, sentence_bigrams_complete = [], [], []
    current_label = ""
    for f in files:
        sentenceLabelList = []
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
        lines = open("../Preprocess/"+f,"rb").readlines()
        # print "length of lines: ", len(lines)
        sent_bigrams, line_wise_bigrams = get_sentence_bigrams(lines)
        # print "length of line_wise_bigrams: ", len(line_wise_bigrams)
        for line in line_wise_bigrams:
            sentenceLabelList.append((line, label))
        sentence_bigrams_complete += sent_bigrams
        sentence_wise_bigrams_complete += sentenceLabelList
        # print "length of sentence_wise_bigrams_complete in loop: ", len(sentence_wise_bigrams_complete)

    frequent_bigrams = keep_frequent_bigrams(sentence_bigrams_complete)
    adjfeatures = open("../Preprocess/stemmedFeatures.txt", "rb").readlines()
    print "length of sentence_wise_bigrams_complete: ", len(sentence_wise_bigrams_complete)
    #print "adjfeatures=",adjfeatures

    # for ff in range(len(tfidffeatures)):
    #     tfidffeatures[ff] = str(tfidffeatures[ff])

    for ff in range(len(adjfeatures)):
        adjfeatures[ff] = adjfeatures[ff].replace("\n","")

    #print "adjfeatures=",adjfeatures
    # print sentence_bigrams_complete[:10]
    finalfeatures = commonFeatures(frequent_bigrams, adjfeatures)
    print "length of common features list: ", len(finalfeatures)
    #print "finalfeatures=",finalfeatures

    X, Y = convertXY(sentence_wise_bigrams_complete, finalfeatures)
    print "length of X: ", len(X)
    zippedXY = zip(X,Y)
    pickle.dump(zippedXY,open("zippedXY_bigrams_10k_lessFeatures.p", "wb"))









