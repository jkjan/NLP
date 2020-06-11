import pickle
import pandas as pd

def get_data(path):
    # read data from path
    data = pd.read_table(path)

    # removing duplicates
    data.drop_duplicates(subset=['document'], inplace=True)

    # removing null data
    data = data.dropna(how='any')

    # removing any characters that are not Korean
    data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    # removing rows that their documents are empty
    data = data[data.document != '']

    return data


def tokenize(data_path, tokenizer, stopwords, save_path):
    data = get_data(data_path)

    try:
        tokenized = pickle.load(open(save_path, "rb"))
        print("tokenized file exists.")

    except FileNotFoundError:
        print("tokenized file doesn't exist. initializing tokenizer")

        # a list of tokenized sentences
        tokenized = []

        # tokenizing data
        for i in range(len(data)):
            # tokenizing a sentence
            tokenized_sent = tokenizer().morphs(data.iloc[i]['document'], stem=True)

            # removing stopwords
            tokenized_sent = [word for word in tokenized_sent if not word in stopwords]

            tokenized.append(tokenized_sent)

        pickle.dump(tokenized, open(save_path, "wb"))


    return data, tokenized