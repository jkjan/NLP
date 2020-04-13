import pickle
import torch

def load_data(path, train=True):
    data_type = "_train" if train else "_test"
    try:
        data = pickle.load(open(path + data_type + ".pkl", "rb"))
    except FileNotFoundError:
        print(("훈련" if train else "테스트") + " 데이터가 없습니다.")
        print("빈 딕셔너리가 반환됩니다.")
        return {}

    return data

def one_hot_encode(size, index):
    one_hot_vector = torch.zeros(size)
    one_hot_vector[index] = 1
    return one_hot_vector

def data_load_test(data, is_list=True):
    if is_list:
        for i in range(5):
            print(data[i])
        print()
        return

    else:
        for (index, key) in enumerate(data):
            if index >= 5:
                break
            print(key, data[key])
        print()
        return