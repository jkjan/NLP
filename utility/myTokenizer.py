from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle

def tokenizeFromFile(readFileDir, saveFileDir, stopwords, lim):
    readFile = open(readFileDir, "r", encoding="UTF8")
    saveFile = open(saveFileDir, "w", encoding="UTF8")
    preprocessed = []
    lemmatizer = WordNetLemmatizer()
    while True:
        line = readFile.readline()
        if len(line) == 0:
            print("File read finished")
            readFile.close()
            break
        sentTokenized = sent_tokenize(line)

        for sent in sentTokenized:
            sent = re.sub("[^a-zA-Z]", " ", sent)
            wordTokenized = word_tokenize(sent)
            i = 0
            while i < len(wordTokenized):
                if len(wordTokenized[i]) <= lim or \
                        wordTokenized[i] in stopwords:
                    wordTokenized.remove(wordTokenized[i])
                else:
                    wordTokenized[i] = wordTokenized[i].lower()
                    wordTokenized[i] = lemmatizer.lemmatize(wordTokenized[i])
                    saveFile.write(wordTokenized[i])
                    if i < len(wordTokenized) - 1:
                        saveFile.write(" ")
                    i += 1

            saveFile.write("\n")
            preprocessed.append(wordTokenized)

    saveFile.close()

    return preprocessed


def tokenizeFromList(readList, saveFileDir, stopwords, lim):
    saveFile = open(saveFileDir, "w", encoding="UTF8")
    preprocessed = []
    lemmatizer = WordNetLemmatizer()
    vocabulary = {}

    print("Tokenizing starts")
    vocabulary_cnt = 0
    word_in_sent_cnt = 0

    for line in readList:
        line = re.sub("<.+?>", " ", line)
        sentTokenized = sent_tokenize(line)

        for sent in sentTokenized:
            sent = re.sub("[^a-zA-Z]", " ", sent)
            wordTokenized = word_tokenize(sent)
            i = 0
            while i < len(wordTokenized):
                if len(wordTokenized[i]) <= lim or \
                        wordTokenized[i] in stopwords:
                    wordTokenized.remove(wordTokenized[i])
                else:
                    wordTokenized[i] = wordTokenized[i].lower()
                    wordTokenized[i] = lemmatizer.lemmatize(wordTokenized[i])
                    word_in_sent_cnt += 1
                    saveFile.write(wordTokenized[i])

                    if wordTokenized[i] not in vocabulary:
                        vocabulary[wordTokenized[i]] = vocabulary_cnt
                        vocabulary_cnt += 1

                    if i < len(wordTokenized) - 1:
                        saveFile.write(" ")
                    i += 1

            if word_in_sent_cnt > 0:
                saveFile.write("\n")
                word_in_sent_cnt = 0

            preprocessed.append(wordTokenized)

    saveFile.close()
    print("Tokenized finished. The tokenized file is saved in", saveFileDir)

    with open(saveFileDir[:-4]+"_vocabulary.pkl", "wb") as dicSave:
        pickle.dump(vocabulary, dicSave)

    with open(saveFileDir[:-4]+".pkl", "wb") as preSave:
        pickle.dump(preprocessed, preSave)

    return preprocessed, vocabulary


def tokenizeFromStr(string, stopwords, lim):
    string = re.sub("[^a-zA-Z]", " ", string)
    preprocessed = []
    wordTokenized = word_tokenize(string)
    lemmatizer = WordNetLemmatizer()
    i = 0

    while i < len(wordTokenized):
        if len(wordTokenized[i]) <= lim or \
                wordTokenized[i] in stopwords:
            wordTokenized.remove(wordTokenized[i])
        else:
            wordTokenized[i] = wordTokenized[i].lower()
            wordTokenized[i] = lemmatizer.lemmatize(wordTokenized[i])
            preprocessed.append(wordTokenized[i])

            i += 1

    return preprocessed