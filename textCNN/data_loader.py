import pandas as pd
from myTokenizer import tokenizeFromList
import pickle

def getMovieData():
    return pd.read_csv("../data/English/"
                       "IMDB Dataset/"
                       "IMDB Dataset.csv",
                       low_memory=False)

def getTokenizedData(data):
    try:
        tokenized = pickle.load(open("../data/English/IMDB Dataset/tokenized.pkl", "rb"))
        vocabulary = pickle.load(open("../data/English/IMDB Dataset/tokenized_vocabulary.pkl", "rb"))

        print("Tokenized word and vocabulary file exist.")

    except FileNotFoundError:
        print("Tokenized word or vocabulary file does not exist.")

        tokenized, vocabulary = tokenizeFromList(
                                    readList=data,
                                    saveFileDir="../data/English/IMDB Dataset/tokenized.txt",
                                    stopwords=[],
                                    lim=1)

    return tokenized, vocabulary

