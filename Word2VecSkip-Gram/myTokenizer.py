from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize, sent_tokenize

def myTokenizer(readFileDir, saveFileDir, stopwords, lim):
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