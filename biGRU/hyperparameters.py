from data_loader import load_data
import torch
import torch.nn as nn
from GRU import GRU


# decide which device will be used
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

# a path to tokenized and pos tagged data of 100 korean lyrics
data_path = "./data/pos_tagged.pkl"

# a path to dictionary data of 100 korean lyrics, which maps a word to a corresponding index
dictionary_path = "./data/word_to_idx.pkl"

# a dictionary that maps a word to a corresponding index
word_to_idx = load_data(dictionary_path)

# data of lyrics
data = load_data(data_path)

# a dictionary that maps an index to a corresponding word
idx_to_word = {v: k for k, v in word_to_idx.items()}

# the number of unique words
input_size = len(word_to_idx)

# the number of hidden layers
hidden_size = 512

# a size of output tensor
output_size = len(word_to_idx)

# the number of layers
num_layers = 1

# a size of batch
batch_size = 1

# learning rate
learning_rate = 0.001

# a deep learning model to use
model = GRU(input_size, hidden_size, output_size, batch_size, device, num_layers)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer with backpropagation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# the number of iteration
n_iter = 100000

# a current epoch
now_epoch = 0

# the number of words to generate or train
seq_len = 5

# print frequency
print_every = 5000

# plot frequency
plot_every = 500