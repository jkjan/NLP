import torch.nn as nn
import torch.nn.init

class SkipGram(nn.Module):
    def __init__(self, vocabulary_size, dimension_size):
        super(SkipGram, self).__init__()

        # the model's hyper parameter
        self.vocabulary_size = vocabulary_size
        self.dimension_size = dimension_size

        # initializing u and v embedding tensor
        self.v = nn.Embedding(vocabulary_size, dimension_size)
        self.u = nn.Embedding(dimension_size, vocabulary_size)

        self.output = nn.Softmax(dim=1)

        init_range = 1.0 / self.dimension_size

        # regularization
        # nn.init.uniform_(self.u.weight.data, -init_range, init_range)
        # nn.init.constant_(self.v.weight.data, 0)
        torch.nn.init.xavier_normal_(self.u.weight)
        torch.nn.init.xavier_normal_(self.v.weight)


    def forward(self, input):
        # print("v :")
        # print(self.v.weight)
        # print()
        # print("u :")
        # print(self.u.weight)
        out = torch.matmul(input, self.v.weight)
        out = torch.matmul(out, self.u.weight)
        out = self.output(out)

        return out
