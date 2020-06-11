from gensim.models.word2vec import Word2Vec


def get_word_embedding(data, save_path):
    try:
        w2v = Word2Vec.load(save_path)
        print("the korean word2vec file exists")

    except FileNotFoundError:
        print("the Korean word2vec file does not exists")

        w2v = Word2Vec(data, size=100, window=5, min_count=1, workers=4, sg=1)
        w2v.save(save_path)


    return w2v
