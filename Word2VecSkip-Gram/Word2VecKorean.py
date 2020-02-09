from gensim.models.word2vec import Word2Vec

model = None
try:
    model = Word2Vec.load("../data/Korean/word2vec/kor_w2v")
    print("the korean word2vec file exists!")
except FileNotFoundError:
    print("the Korean word2vec file does not exists T.T")
    file = None
    try:
        file = open("../data/Korean/tokenized/corpus_mecab.txt", "r", encoding="UTF8")
        print("the text file exists!")
    except IOError:
        print("the text file does not exists! T.T")
        exit()
    corpus = []
    while True:
        line = file.readline()
        if not line:
            print("file read finished")
            break
        corpus.append(line.split(" "))

    model = Word2Vec(corpus, size=100, window=5, min_count=5, workers=4, sg=0)
    model.save("../data/Korean/word2vec/kor_w2v")

def vectorSubtract(a, b, c):
    result = model.wv.most_similar(positive=[c, b], negative=[a])
    return result[0][0]

modelResult1 = vectorSubtract("자동차", "기계", "나비")
print(modelResult1)
