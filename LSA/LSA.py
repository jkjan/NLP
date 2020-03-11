import re
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

cnnTF_IDF = None
vocabs = None

vectorizer = TfidfVectorizer(max_features=1000, lowercase=True, stop_words="english", max_df=0.3)

try:
    cnnTF_IDF = pickle.load(open("cnn-tf-idf.pkl", "rb"))
    vocabs = pickle.load(open("cnn-terms.pkl", "rb"))
    print("There is a tf-idf file")
except FileNotFoundError:
    print("No tf-idf file exists.")
    cnnFile = open("../data/English/raw/cnn/all.txt", "r", encoding='UTF8')
    cnnDoc = []
    cnnTokenized = []
    while True:
        line = cnnFile.readline()
        if len(line) == 0:
            print("File read finished")
            break
        line = re.sub("[^a-zA-Z ]", "", line)
        tokenized = word_tokenize(line)
        for word in tokenized:
            if word == "CNN" or word == "highlight":
                tokenized.remove(word)
        cnnTokenized.append(tokenized)

        sent = ""
        for w in tokenized:
            sent += (w + " ")
        cnnDoc.append(sent)

    cnnTF_IDF = vectorizer.fit_transform(cnnDoc)
    with open("cnn-tf-idf.pkl", 'wb') as handle:
        pickle.dump(cnnTF_IDF, handle)

    vocabs = vectorizer.get_feature_names()
    with open("cnn-terms.pkl", 'wb') as handle:
        pickle.dump(vectorizer.get_feature_names(), handle)

svd = TruncatedSVD(n_components=8)
svd.fit(cnnTF_IDF)
topics = svd.components_

for index, topic in enumerate(topics):
    print("Topic %d : " % (index+1), end="")
    print([(vocabs[i], topic[i].round(5)) for i in topic.argsort()[:-6:-1]])
