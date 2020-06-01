from data_loader import *
import torch.nn as nn
from GRU import GRU
import sys
import matplotlib.pyplot as plt
import time
import math

device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available!")
    torch.backends.cudnn.benchmark = True
    print('Memory Usage:')
    print('Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    print('cuDNN:    ', torch.backends.cudnn.version())

else:
    device = torch.device("cpu")


# 시 품사 태깅 경로
data_path = "../data/Korean/processed/lyrics/tokenized.pkl"

# 시 딕셔너리 경로
dictionary_path = "../data/Korean/processed/lyrics/dict.pkl"

word_to_idx = load_data(dictionary_path)
data = load_data(data_path)


# 단어를 인덱스로 매핑해주는 딕셔너리
idx_to_word = {v: k for k, v in word_to_idx.items()}

# 입력 사이즈, 즉 나올 수 있는 단어 가지의 수
input_size = len(word_to_idx)

# 은닉층 수
hidden_size = 512

# 출력 사이즈
output_size = len(word_to_idx)

# 레이어의 수
num_layers = 1

# 배치 크기
batch_size = 10

# 학습률
learning_rate = 0.1

# 사용할 모델
model = GRU(input_size, hidden_size, output_size, batch_size, device, num_layers).to(device)

# 손실 함수의 기준
criterion = nn.CrossEntropyLoss()

# 최적화 도구
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 반복 횟수
n_iter = 1000

# 현재 문장 위치
now_epoch = 0

# 생성할 단어 수, 한 번에 학습할 단어 수
seq_len = 5

torch.autograd.set_detect_anomaly(True)


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
        one_hot_vector = torch.zeros(seq_len, batch_size, input_size).to(device)
        for i, doc in enumerate(docs):
            for j in range(0, seq_len):
                try:
                    word = doc[now_word + j]
                    one_hot_vector[j][i][word_to_idx[word]] = 1
                    flag = True
                except IndexError:
                    break

        target.append(one_hot_vector)
        now_word += 1

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


def train(target, label):
    # 은닉층 초기화
    hidden = model.init_hidden()
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


losses = []
cur_loss = 0

print_every = n_iter/10
plot_every = n_iter/10
start = time.time()

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