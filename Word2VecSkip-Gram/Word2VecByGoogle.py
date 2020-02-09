import gensim

model = None
try:
    model = gensim.models.KeyedVectors.load_word2vec_format('../data/English/word2vec_by_google/GoogleNews-vectors-negative300.bin'
                                                        , binary=True)
except FileNotFoundError:
    print("File Not Found")
    exit()

def vectorSubtract(a, b, c):
    result = model.most_similar(positive=[c, b], negative=[a])
    return result[0][0]

print(vectorSubtract("man", "king", "woman"))

