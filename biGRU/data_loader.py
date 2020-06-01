import pickle

def load_data(path):
    """
    :param path: pkl 파일 경로
    :return: 데이터
    """
    try:
        data = pickle.load(open(path, "rb"))
    except FileNotFoundError:
        print("데이터가 없습니다.")
        print("빈 딕셔너리가 반환됩니다.")
        return {}

    return data


def data_load_test(data, is_list=True):
    """
    :param data: 데이터
    :param is_list: 리스트 여부
    :return: x
    """
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