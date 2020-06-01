import pandas as pd

# 본인 경로로 바꿔서 확인
tsv_path = "../data/Korean/original/kwn_synset_list.tsv"

df = pd.read_csv(tsv_path, sep='\t',  encoding="utf8")

# 유의어 리스트 반환
def get_synonyms(word):
    synonyms = set()
    for cand in df['korean_lemmas']:
        if word in cand:
            synonym_list = cand.split(", ")
            for synonym in synonym_list:
                synonyms.add(synonym)

    if len(synonyms) == 0:
        print("유의어가 없습니다.")
    return synonyms


print(get_synonyms("만족"))