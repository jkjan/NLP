import torch.nn as nn
from SkipGram import SkipGram
from data_loader import *
import torch
from Batch import Batch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Now you are using", device + "!")

# getting data
# total 50k reviews, each is positive or negative
movieData = getMovieData()

# a tokenized sentence (and each sentence is word tokenized) list
# and a dictionary of unique words as keys and their index as values
tokenized, vocabulary = getTokenizedData(movieData["review"])

# the number of words and dimension and a size of window
vocabulary_size = len(vocabulary)
dimension_size = 100
window = 4
learning_rate = 0.001
batch_size = 100

# Total 50000 reviews,
# composed of 616273 words and 89858 unique words.
print("Total", len(movieData["review"]), "reviews,")
print("composed of", len(tokenized), "words and", len(vocabulary), "unique words.")

model = SkipGram(vocabulary_size, dimension_size).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training will be initiated with")
print(model)

print()
print("-----Training starts-----")
print()

word_cnt = 0
avg_loss = 0
epoch_cnt = 0

def get_batch(epoch):
    empty = torch.Tensor()

    # input batch
    for i in range(epoch, epoch+batch_size):

        adding = index_to_one_hot(index).to(device)
        empty = torch.cat([empty, adding], 0)

    # output batch




for sentence in tokenized:
    for i, word in enumerate(sentence):
        # getting indices of context words around the center word (which is answers)
        answer_label = get_label(sentence, i)
        if len(answer_label) == 0:
            continue

        # an index corresponding a center word
        index = vocabulary[word]

        # a one-hot vector corresponding the index
        input = index_to_one_hot(index).to(device)

        # initializing a gradient
        optimizer.zero_grad()

        # training a model
        expectation = model(input)

        # concatenating output vector (shape of 1xV) as many as a window size,
        # which becomes a vector size of (window x V)
        output_batch = torch.cat(len(answer_label) * [expectation], 0)

        # calculating a loss function
        loss = criterion(output_batch, answer_label)

        # back propagation
        loss.backward()

        word_cnt += 1
        avg_loss += loss.item() / 100

        if word_cnt == 100:
            epoch_cnt += 1
            print("[epoch {}] : {}".format(epoch_cnt, avg_loss))
            word_cnt = 0
            avg_loss = 0

        # adjusting weight matrices (= word embedding look-up table u and v)
        optimizer.step()

print("Training finished.")
torch.save(model.state_dict(), "./model.pth")