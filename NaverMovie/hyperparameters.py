from konlpy.tag import Okt
from torch.optim.lr_scheduler import StepLR

from preprocess import tokenize
from word2vec_embed import get_word_embedding
import torch
import torch.nn as nn
from GRU import GRU


# paths, names of needed data
path_to_data = "../data/Korean/original/naver_movie/"
path_to_tokenized = "../data/Korean/tokenized/naver_movie/"
path_to_w2v = "../data/Korean/word2vec/"

train_file = "ratings_train.txt"
test_file = "ratings_test.txt"
w2v_file = "naver_w2v.w2v"
tokenized_train_file = "tokenized_train.pkl"
tokenized_test_file = "tokenized_test.pkl"

tokenizer = Okt
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

train_data, tokenized_train = tokenize(path_to_data + train_file,
                                       tokenizer,
                                       stopwords,
                                       path_to_tokenized + tokenized_train_file)

test_data, tokenized_test = tokenize(path_to_data + test_file,
                                     tokenizer,
                                     stopwords,
                                     path_to_tokenized + tokenized_test_file)

test_data = test_data[:200]
tokenized_test = tokenized_test[:200]

w2v = get_word_embedding(tokenized_train + tokenized_test, path_to_w2v + w2v_file)

# a size of batch
batch_size = 1

# learning rate
learning_rate = 0.001


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available!")
    torch.backends.cudnn.benchmark = True
    print('Memory Usage:')
    print('Max Alloc:', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    print('cuDNN:    ', torch.backends.cudnn.version())

else:
    device = torch.device("cpu")


# the number of unique words
input_size = 100

# the number of hidden layers
hidden_size = 512

# a size of output tensor
output_size = 2

# the number of layers
num_layers = 2

# a deep learning model to use
model = GRU(input_size, hidden_size, output_size, batch_size, device, num_layers)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer with backpropagation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=300, gamma=0.1)

# print frequency
print_every = 50

# plot frequency
plot_every = 50

# the number of iteration
n_iter = 1000

# a current epoch
now_epoch = 0