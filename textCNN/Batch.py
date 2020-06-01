import torch
from utils import index_to_one_hot, get_label
from train import batch_size, tokenized, vocabulary

class Batch:
    def __init__(self):
        self.sentence = 0
        self.word_in_sentence = 0

    def get_batch(self):
        input = torch.Tensor()
        output = torch.Tensor()

        for i in range(0, batch_size):
            if self.word_in_sentence == len(tokenized[self.sentence]):
                self.sentence += 1
                self.word_in_sentence = 0

            index = vocabulary[tokenized[self.sentence][self.word_in_sentence]]
            adding = index_to_one_hot(index)
            input = torch.cat([input, adding], 0)

            get_label(tokenized[self.sentence], self.word_in_sentence)


