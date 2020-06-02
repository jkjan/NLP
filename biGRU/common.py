import time
import sys
import math
from hyperparameters import *
import random


def lyric_to_tensor(lyric):
    """
    :param lyric: a piece of lyric in data
    :return: one-hot encoded tensor of the lyric
    """
    tensor = torch.zeros(len(lyric), 1, input_size).to(device)
    for i in range(len(lyric)):
        word = lyric[i]
        tensor[i][0][word_to_idx[word]] = 1
    return tensor


def target_to_tensor(lyric):
    """
    :param lyric: a piece of lyric in data
    :return: indices of words in lyric
    """
    target_indices = [word_to_idx[lyric[i]] for i in range(1, len(lyric))]
    target_indices.append(input_size - 1) # EOS
    return torch.LongTensor(target_indices).to(device)


def random_training_example():
    """
    :return: random pair of lyric
    """
    rand_lyric = data[random.randint(0, len(data) - 1)]
    return lyric_to_tensor(rand_lyric), target_to_tensor(rand_lyric)


def train(lyrics, target):
    target.unsqueeze_(-1)
    hidden = model.init_hidden()
    optimizer.zero_grad()
    loss = 0

    for i in range(lyrics.size(0)):
        input = lyrics[i].reshape((1, 1, input_size))
        # GRU 출력
        output, hidden = model(input, hidden)

        # 손실 계산
        cur_loss = criterion(output, target[i]).to(device)
        loss += cur_loss

    # backpropagation
    loss.backward()
    optimizer.step()
    # for p in model.parameters():
    #     p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / lyrics.size(0)


def make_batch(docs):
    """
    :param docs: 배치로 묶을 문장들
    :return: 배치로 묶인 원-핫 인코딩된 문장. seq_len 씩 끊어져 구분된 리스트,
    size = (학습할 단어 수, 배치 크기, 입력 크기)
    """
    target = []
    now_word = 0
    flag = True

    while flag:
        flag = False
        one_hot_vector = torch.zeros(seq_len, batch_size, input_size)
        for i, doc in enumerate(docs):
            for j in range(0, seq_len):
                try:
                    word = doc[now_word + j]
                    one_hot_vector[j][i][word_to_idx[word]] = 1
                    flag = True
                except IndexError:
                    break
        target.append(one_hot_vector.to(device))
        now_word += seq_len

    return target


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def print_string(string, j):
    for k in range(0, seq_len):
        expected = string[k][j].argmax(0)
        sys.stdout.write(idx_to_word[expected.item()] + " ")


def generate(word):
    """
    :param word: starting word
    :return: a lists of generated words
    """
    with torch.no_grad():
        input_tensor = lyric_to_tensor(word)
        hidden = model.init_hidden()

        output_lyric = ""

        for i in range(20):
            output, hidden = model(input_tensor, hidden)

            topv, topi = output.topk(1)
            topi = topi[0][0].item()

            if topi == input_size - 1:
                break
            else:
                word = idx_to_word[topi]
                output_lyric += (word + " ")

            input_tensor = lyric_to_tensor([word])

        return output_lyric
