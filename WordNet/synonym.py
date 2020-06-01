import pickle
from konlpy.tag import Kkma, Okt, Komoran

class_index = ["기쁨", "신남", "만족", "슬픔", "분노", "피로"]

def morph_tokenize(tokenizer):
    cnt = 0

    # 본인 이름
    my_name = "jeongin"

    # 본인 경로
    my_path = "../data/Korean/processed/"

    dict_path = my_path + my_name + "_dictionary"

    if tokenizer == Kkma:
        tokenizer_type = "_Kkma"

    elif tokenizer == Okt:
        tokenizer_type = "_Okt"

    else:
        tokenizer_type = "_Komoran"

    try:
        emotion_dictionary = pickle.load(open(dict_path + tokenizer_type + ".pkl", "rb"))

    except FileNotFoundError:
        emotion_dictionary = {}

        input = open(my_path + my_name + "_synonyms.txt", encoding="utf8")

        while True:
            line = input.readline()
            if len(line) == 0:
                break
            cnt += 1

            if cnt % 2 == 0:
                extracted = tokenizer().morphs(line)

                try:
                    extracted.remove("\n")
                except ValueError:
                    pass

                emotion_dictionary[class_index[int((cnt - 1) / 2)]] = set(extracted)
            else:
                continue

        with open(dict_path + tokenizer_type + ".pkl", "wb") as dict_save:
            pickle.dump(emotion_dictionary, dict_save)

        input.close()

        output = open(my_path + my_name + "_morphs" + tokenizer_type + ".txt", "wt", encoding="utf8")

        for index, (key, value) in enumerate(emotion_dictionary.items()):
            output.write(key + "\n")
            for j, word in enumerate(value):
                output.write(word)
                if j == len(value) - 1:
                    output.write("\n")
                else:
                    output.write(" ")

        output.close()

    return emotion_dictionary


print("Okt")
print(morph_tokenize(Okt), end="\n\n")

print("Kkma")
print(morph_tokenize(Kkma), end="\n\n")

print("Komoran")
print(morph_tokenize(Komoran), end="\n\n")