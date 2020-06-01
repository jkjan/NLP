import time
import sys
import math
from hyperparameters import *

def encode_word(word):
    """
    :param word: 단어
    :return: 원-핫 인코딩된 단어, size = (1, 입력 크기)
    """
    one_hot_vector = torch.zeros(1, input_size)
    one_hot_vector[word_to_idx[word]] = 1
    return one_hot_vector.to(device)


def encode_doc(doc):
    """
    :param doc: 문장
    :return: 원-핫 인코딩된 문장, size = (문장의 길이, 1, 입력 크기)
    """
    one_hot_vector = torch.zeros(len(doc), 1, input_size)
    for i, word in enumerate(doc):
        one_hot_vector[i][0][word_to_idx[word]] = 1
    return one_hot_vector.to(device)


def lyric_to_tensor(lyric):
    tensor = torch.zeros(len(lyric), 1, input_size).to(device)
    for i in range(len(lyric)):
        word = lyric[i]
        tensor[i][0][word_to_idx[word]] = 1
    return tensor


def target_to_tensor(lyric):
    target_indices = [word_to_idx[lyric[i]] for i in range(1, len(lyric))]
    target_indices.append(input_size - 1)
    return torch.LongTensor(target_indices).to(device)


def train(target, label):
    # 은닉층 초기화
    hidden = model.init_hidden(device)
    optimizer.zero_grad()
    loss = 0

    for i in range(len(target)):
        # GRU 출력
        output, hidden = model(target[i], hidden)
        (seq, bat, inp) = output.size()
        output = output.reshape(seq, inp, bat)

        # 손실 계산
        l = criterion(output, label[i].argmax(2)).to(device)
        loss += l

        # for j in range(0, batch_size):
        #     print_string(target[i], j)
        #     sys.stdout.write(" -> ")
        #     print_string(output.reshape(seq, bat, inp), j)     # 생성된 문자열
        #     sys.stdout.write(" / ")
        #     print_string(label[i], j)
        #     sys.stdout.write("\n")
        # sys.stdout.write("\n")

    # 역전파 및 변수 조정
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


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
