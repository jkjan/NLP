import re
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

model = None
try:
    model = Word2Vec.load('../data/English/word2vec/cnn_w2v')
    print("the word2vec file exists!")

    tokenizedLoadFile = open("../data/English/tokenized/cnn.txt", "r", encoding="UTF8")
    print("tokenized text exists.")
    cnnPreprocessed = tokenizedLoadFile
except IOError or FileNotFoundError:
    print("the word2vec file does not exist! T.T")

    print("tokenized text doesn't exist.")
    file = open("../data/English/raw/cnn/all.txt", "r", encoding='UTF8')
    tokenizedSaveFile = open("../data/English/tokenized/cnn.txt", "w", encoding="UTF8")

    cnnPreprocessed = []
    lemmatizer = WordNetLemmatizer()
    stopWords = stopwords.words("english")
    while True:
        line = file.readline()
        if len(line) == 0:
            print("File read finished")
            break
        cnnSentTokenized = sent_tokenize(line)
        # 문장별로 끊음

        for sent in cnnSentTokenized:
            sent = re.sub("[^a-zA-Z]", " ", sent)
            cnnWordTokenized = word_tokenize(sent)
            i = 0
            while i < len(cnnWordTokenized):
                if cnnWordTokenized[i] in ["CNN", "highlight"] or \
                        len(cnnWordTokenized[i]) <= 2 or \
                        cnnWordTokenized[i] in stopWords:
                    cnnWordTokenized.remove(cnnWordTokenized[i])
                else:
                    cnnWordTokenized[i] = cnnWordTokenized[i].lower()
                    cnnWordTokenized[i] = lemmatizer.lemmatize(cnnWordTokenized[i])
                    tokenizedSaveFile.write(cnnWordTokenized[i])
                    if i < len(cnnWordTokenized) - 1:
                        tokenizedSaveFile.write(" ")
                    i += 1

            tokenizedSaveFile.write("\n")
            cnnPreprocessed.append(cnnWordTokenized)

    tokenizedSaveFile.close()

    print("Word2Vec Train initiating")
    model = Word2Vec(sentences=cnnPreprocessed, size=100, window=5, min_count=5, workers=4, sg=0)
    # size = 워드 벡터의 차원, N에 해당.
    # window = 문맥 윈도우 크기 (주변 몇 단어?)
    # min_count = 단어 최소 빈도 수 제한 (빈도 적은 단어는 학습 x)
    # workers = 학습을 위한 프로세스 수
    # sg 0은 cbow, 1은 skip-gram
    print("Word2Vec Train completed")
    model.save('../data/English/word2vec/cnn_w2v')
    print("Word2Vec file saved")


def vectorSubtract(a, b, c):
    result = model.wv.most_similar(positive=[c, b], negative=[a])
    return result[0][0]

mostSimilar = model.wv.most_similar("kid")
for i in range(0, 6):
    print(mostSimilar[i])