import torch
from trainer import input_size, word_to_idx, batch_size, seq_len, device
import time
import math

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