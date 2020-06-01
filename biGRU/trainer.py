import sys
import matplotlib.pyplot as plt
from utils import random_training_example, time_since, train, lyric_to_tensor, target_to_tensor
from hyperparameters import *
import time

# a list of losses
losses = []

# current loss
loss = 0

# sum of losses for an epoch
total_loss = 0

# now!
start = time.time()

for i in range(1, n_iter + 1):
    output, loss = train(*random_training_example())
    total_loss += loss

    if i % print_every == 0:
        sys.stdout.write("%d %d%% (%s) %.4f\n" % (i, i / n_iter * 100, time_since(start), loss))

    if i % plot_every == 0:
        losses.append(total_loss/plot_every)
        total_loss = 0


plt.figure()
plt.plot(losses)
plt.show()

torch.save(model, "gru.pt")