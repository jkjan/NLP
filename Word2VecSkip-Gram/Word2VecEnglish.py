import re
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec

model = None
try:
    model = Word2Vec.load('../data/English/word2vec/eng_w2v')
    print("the word2vec file exists!")
except IOError:
    print("the word2vec file does not exist! T.T")
    targetXML = open('../data/English/raw/ted_en-20160408.xml', 'r', encoding='UTF8')
    targetText = etree.parse(targetXML)
    parseText = '\n'.join(targetText.xpath('//content/text()'))
    # xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.

    contentText = re.sub(r'\([^)]*\)', '', parseText)
    # 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
    # 해당 코드는 괄호로 구성된 내용을 제거.

    sentText = sent_tokenize(contentText)
    # 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.

    normalizedText = []
    for string in sentText:
         tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
         normalizedText.append(tokens)
    # 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.

    result = []
    result = [word_tokenize(sentence) for sentence in normalizedText]
    file = open("../data/English/tokenized/ted.txt", "w")
    for tokenized in result:
        for word in tokenized:
            file.write(word+" ")
    file.close()
    # 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.

    model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)
    # size = 워드 벡터의 차원.
    # window = 문맥 윈도우 크기 (주변 몇 단어?)
    # min_count = 단어 최소 빈도 수 제한 (빈도 적은 단어는 학습 x)
    # workers = 학습을 위한 프로세스 수
    # sg 0은 cbow, 1은 skip-gram
    model.save('../data/English/word2vec/eng_w2v')

def vectorSubtract(a, b, c):
    result = model.wv.most_similar(positive=[c, b], negative=[a])
    return result[0][0]

print(vectorSubtract("man", "king", "woman"))