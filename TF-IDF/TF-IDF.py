import numpy as np
import pandas as pd

data = {
    '나는' : np.array([1,0,1,0]),
    '쇼팽' : np.array([1,0,0,0]),
    '어제' : np.array([0,1,0,0]),
    '음악' : np.array([1,1,1,2]),
    '을(를)' : np.array([1,0,1,0]),
    '은' : np.array([0,0,0,1]),
    '역시' : np.array([0,0,0,1]),
    '클래식' : np.array([0,0,0,1]),
    '시간' : np.array([0,1,0,0]),
    '잤어요' : np.array([0,1,0,0]),
    '듣는' : np.array([0,0,1,0]),
    '것을' : np.array([0,0,1,0]),
    '좋아합니다' : np.array([0,0,1,0])
}

tdm = pd.DataFrame.from_dict(data, orient="index")
dm = np.array(np.sum([tf > 0 for tf in tdm.values], axis=1))[:, np.newaxis]
idm = np.log(len(tdm)/(1+dm))
tf_idm = pd.DataFrame((tdm.values * dm), columns=['문장1', '문장2', '문장3', '문장4'], index=data.keys())
print(tf_idm)