import time
import sys
import math
from hyperparameters import *
import random
import torch


def word_to_tensor(word):
    """
    :param word: a word in a document
    :return: a tensor of the word, size of (1, input_size)
    """
    vector = w2v.wv[word]
    tensor = torch.from_numpy(vector)
    tensor = tensor.unsqueeze_(0)

    return tensor


def doc_to_tensor(doc):
    """
    :param doc: a tokenized list of a document
    :return: tensor of the doc, size of (len(doc), 1, input_size)
    """
    tensor = torch.cat([word_to_tensor(word) for word in doc], 0).to(device)
    tensor = tensor.unsqueeze_(1)

    return tensor


def target_to_tensor(target):
    tensor = torch.tensor(target).to(device).unsqueeze_(-1)
    return tensor


def random_training_example():
    """
    :return: random training example (tensor(len(doc), 1, input_size), target index)
    """
    while True:
        rand_sample = random.randint(0, len(train_data) - 1)
        if len(tokenized_train[rand_sample]) != 0:
            break

    return doc_to_tensor(tokenized_train[rand_sample]), target_to_tensor(train_data.iloc[rand_sample]['label'])


def train(input, target):
    hidden = model.init_hidden()
    optimizer.zero_grad()

    for i in range(input.size(0)):
        resized = input[i].reshape(1, 1, input_size)
        output, hidden = model(resized, hidden)

    loss = criterion(output, target).to(device)

    loss.backward()
    optimizer.step()
    scheduler.step()

    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def predict(input, target):
    with torch.no_grad():
        hidden = model.init_hidden()

        for i in range(input.size(0)):
            resized = input[i].reshape(1, 1, input_size)
            output, hidden = model(resized, hidden)

        topv, topi = output.topk(1)
        topi = topi[0][0].item()

    return topi == target


def accuracy_test():
    total, correct = 0, 0

    for i in range(len(test_data)):
        if len(tokenized_test[i]) > 0:
            total += 1
            answer = predict(doc_to_tensor(tokenized_test[i]), test_data.iloc[i]['label'])
            if answer:
                correct += 1

    return correct/total