import sys
import matplotlib.pyplot as plt
from utils import make_batch, time_since, train
from hyperparameters import *


for iter in range(1, n_iter + 1):
    now_epoch = 0
    sys.stdout.write("iter : " + str(iter) + "\n")

    while now_epoch + batch_size <= len(data):
        target = make_batch(data[now_epoch:now_epoch+batch_size])
        label = target[1:] + [torch.cat((target[-1][1:seq_len], torch.zeros(1, batch_size, input_size).to(device)), 0)]

        output, loss = train(target, label)
        cur_loss += loss

        now_epoch += batch_size

    sys.stdout.write("loss : " + str(loss) + "\n")

    if iter % print_every == 0:
        sys.stdout.write("%d %d%% (%s) %.4f\n" % (iter, iter/n_iter*100, time_since(start), loss))

    if iter % plot_every == 0:
        losses.append(cur_loss/plot_every)
        cur_loss = 0

    sys.stdout.write("\n")

plt.figure()
plt.plot(losses)
plt.show()

torch.save(model, "gru.pt")