import pickle

def load_data(path):
    """
    :param path: a path to a pkl file
    :return: data in the file
    """
    try:
        data = pickle.load(open(path, "rb"))
    except FileNotFoundError:
        print("데이터가 없습니다.")

        return {}

    return data


def data_load_test(data, is_list=True):
    """
    :param data: data
    :param is_list: if a type of the data is list
    :return: None
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