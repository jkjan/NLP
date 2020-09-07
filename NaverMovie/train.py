import sys
import matplotlib.pyplot as plt
import time
from utils import random_training_example, train, time_since, accuracy_test
from hyperparameters import *

is_train = True


if is_train:
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
        output, loss = train(*random_training_example())
        total_loss += loss

        if i % print_every == 0:
            avg_loss = total_loss / plot_every
            sys.stdout.write("%d %d%% (%s) %.4f\n" % (i, i / n_iter * 100, time_since(start), avg_loss))
            losses.append(avg_loss)
            total_loss = 0

    plt.figure()
    plt.plot(losses)
    plt.show()

    sys.stdout.write("train finished\n")
    torch.save(model, "naver_gru.pt")

else:
    model = torch.load("naver_gru.pt")

accuracy = accuracy_test()
print("%.2f%%" % accuracy)