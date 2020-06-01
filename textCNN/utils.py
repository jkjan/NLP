import torch
from train import vocabulary_size, window, vocabulary, device

def get_label(sentence, index):
    output = []
    currentlyLooking = index - window
    maximumWindow = index + window
    while currentlyLooking <= maximumWindow:
        if currentlyLooking == len(sentence):
            break
        elif currentlyLooking != index and currentlyLooking >= 0:
            context_word = sentence[currentlyLooking]
            index_in_vocabulary = vocabulary[context_word]
            output.append(index_in_vocabulary)

        currentlyLooking += 1

    return torch.Tensor(output).long().to(device)


def index_to_one_hot(index):
    one_hot_vector = torch.zeros(vocabulary_size)
    one_hot_vector[index] = 1
    one_hot_vector = one_hot_vector.reshape(1, one_hot_vector.shape[0])
    return one_hot_vector
