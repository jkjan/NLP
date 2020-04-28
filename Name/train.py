from data import get_name_dict, n_letters
import time
import math
from RNN import RNN
from utils import *
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_category, category_lines = get_name_dict()

n_hidden = 128
n_categories = len(category_lines)

rnn = RNN(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()

def train(n_categories, n_hidden, category_tensor, line_tensor, learning_rate=0.005):
    # optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    hidden = rnn.init_hidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    # optimizer.step()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example(all_category, category_lines)
    output, loss = train(n_categories, n_hidden, category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = category_from_output(all_category, output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print("%d %d%% (%s) %.4f %s / %s %s" % (iter, iter/n_iters*100, time_since(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss/plot_every)
        current_loss = 0

plt.figure()
plt.plot(all_losses)
plt.show()