import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv("songdata.csv", low_memory=False).head(20000)

cosine_sim = None
try:
    cosine_sim = np.load("song_processed.npy")
except IOError:
    data['text'] = data['text'].fillna('')
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_marix = tfidf.fit_transform(data['text'])

    cosine_sim = linear_kernel(tfidf_marix, tfidf_marix)
    np.save("song_processed",cosine_sim)

songIndices = pd.Series(data.index, index=data['song']).drop_duplicates()
artistIndices = pd.Series(data.index, index=data['artist']).drop_duplicates()

def recommendMeSimilarTo(artist, title, cosine_sim = cosine_sim):
    artistIndex = artistIndices[artist]
    songIndex = songIndices[title]

    index = None
    if songIndex.size > 1:
        for i in songIndex:
            index = artistIndex[artistIndex == i]
            if index.size > 0:
                break
        index = index.values[0]
    else:
        index = songIndex

    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x : x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    song_indices = [i[0] for i in sim_scores]
    result = data.iloc[song_indices]
    return result

myRecommendation = recommendMeSimilarTo("Metallica", "Master Of Puppets")

for i in range(0, len(myRecommendation)):
    print(myRecommendation['artist'].values[i], "-", myRecommendation['song'].values[i])