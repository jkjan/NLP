from data_loader import *
import torch.nn as nn
from GRU import GRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시 품사 태깅 경로
data_path = "../data/Korean/processed/100_korean_lyrics_pos_tagged"

# 시 딕셔너리 경로
dictionary_path = "../data/Korean/processed/100_korean_lyrics_dictionary"

dictionary_train = load_data(dictionary_path, train=True)
dictionary_test = load_data(dictionary_path, train=False)
data_train = load_data(data_path, train=True)
data_test = load_data(data_path, train=False)

# 알맞은 데이터 가져왔는지 테스트
# data_load_test(data_train)
# data_load_test(data_test)
# data_load_test(dictionary_train, is_list=False)
# data_load_test(dictionary_test, is_list=False)

def train(input_dim, hidden_dim, output_dim, num_layers, learning_rate, batch_size, epoch=5, model_type="GRU"):
    model = GRU(input_dim, hidden_dim, output_dim, num_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(len(data_train)):
        h = model.init_hidden(batch_size, device)


