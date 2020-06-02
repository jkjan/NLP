import sys
import matplotlib.pyplot as plt
import common as cm
import time
import torch

is_train = True

if is_train:
    from hyperparameters import *

    # a list of losses
    losses = []

    # current loss
    loss = 0

    # sum of losses for an epoch
    total_loss = 0

    # now!
    start = time.time()

    print()
    print("----------------")
    print("training starts!")
    print("----------------")
    print()

    for i in range(1, n_iter + 1):
        output, loss = cm.train(*cm.random_training_example())
        total_loss += loss

        if i % print_every == 0:
            sys.stdout.write("%d %d%% (%s) %.4f\n" % (i, i / n_iter * 100, cm.time_since(start), loss))

        if i % plot_every == 0:
            losses.append(total_loss / plot_every)
            total_loss = 0

    plt.figure()
    plt.plot(losses)
    plt.show()

    torch.save(model, "gru.pt")

else:
    model = torch.load("gru.pt")

output_lyrics = cm.generate(["너"])

print(output_lyrics)