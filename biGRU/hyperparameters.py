from data_loader import load_data
import torch
import torch.nn as nn
from GRU import GRU
from time import time

# 시 품사 태깅 경로
data_path = "./data/pos_tagged.pkl"

# 시 딕셔너리 경로
dictionary_path = "./data/word_to_idx.pkl"

# 단어 - 인덱스
word_to_idx = load_data(dictionary_path)

# 데이터
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
model = GRU(input_size, hidden_size, output_size, batch_size, num_layers)

# 손실 함수의 기준
criterion = nn.CrossEntropyLoss()

# 최적화
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 반복 횟수
n_iter = 1000

# 현재 문장 위치
now_epoch = 0

# 생성할 단어 수, 한 번에 학습할 단어 수
seq_len = 5


# 현재 시각
start = time()

# 손실 값 리스트
losses = []

# 현재 손실 값
cur_loss = 0

# 출력 빈도
print_every = n_iter/10

# 플롯 빈도
plot_every = n_iter/10

# 학습 준비
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