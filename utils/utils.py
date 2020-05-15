import pandas as pd
import numpy as np
from collections import Counter
from qcloud_cos_py3.qcloud_cos3.cos_client import CosConfig
from qcloud_cos_py3.qcloud_cos3.cos_client import CosS3Client
from qcloud_cos_py3.qcloud_cos3.cos_auth import CosS3Auth
from qcloud_cos_py3.qcloud_cos3.cos_comm import mapped, get_content_md5

def encodingNumFre (L, topK = 100, isProNaN = False) :
    encodeL = []
    unkNum = 0;
    id2index = {}
    if isinstance(L, pd.core.series.Series):
        L = L.tolist()
    else :
        raise  Exception("The input must be list or pandas series! please check your input!")
    if isProNaN :
        count = [['UNK', -1]]
    else :
        count = []
    if topK :
        count.extend(Counter(L).most_common(topK))
    else :
        count.extend(Counter(L).most_common())
    for idx, _ in count:
        id2index[idx] = len(id2index)
    for item in L :
        if item in id2index :
            encodeL.append(id2index[item])
        else :
            encodeL.append(0)
            unkNum = unkNum  + 1
    if isProNaN :
        count[0][1] = unkNum
    index2id = dict(zip(id2index.values(), id2index.keys()))
    print("Process Finished!")
    return encodeL, count, id2index, index2id


def getVector(index, weight):
    if index in weight.vocab.keys():
        return weight.get_vector(index)
    else:
        return np.array([0.0] * len(weight.vectors[0]))


def getUserAvgSeq(seq, weight):
    res = []
    vecSum = 0
    for item in seq:
        num = 0;
        for indx in item:
            num = num + 1
            vec = getVector(indx, weight)
            vecSum = vecSum + vec
        vecSum = vecSum / num
        res.append(vecSum)
    return res

def getUserMaxSeq(seq,weight):
    res = []
    for item in seq:
        vecSum = []
        for indx in item:
            vec = weight[indx]
            vecSum.append(vec)
        res.append(np.max(np.array(vecSum),axis=0))
    return res

def encode (L,topK = None, isProNaN = False) :
    id2index = {}
    index2id = {}
    length = len(L)
    input  = []
    if length == 1:
        input = L[0]
    else:
        for i in range(length) :
            input.extend(L[i])

    if isProNaN :
        count = [['UNK', -1]]
    else :
        count = []
    if topK :
        count.extend(Counter(input).most_common(topK))
    else :
        count.extend(Counter(input).most_common())
    for idx, _ in count:
        id2index[idx] = len(id2index) + 1
    index2id = dict(zip(id2index.values(), id2index.keys()))
    return id2index, index2id

def transform (L , id2index) :
    res = []
    for item in L :
        if item in id2index :
            res.append(id2index[item])
        else :
            res.append(0)
    return res

def getSeq (df,key="",val="",isSort=False,sortKey="") :

    df_log = df.groupby([key])[val].agg(lambda x: list(x))
    """
    if isSort :
        df_log = df_log.apply(lambda _df: _df.sort_values(by=[sortKey]))
        df_user_seq = df_log.groupby([key])[val].agg(lambda x: list(x))
    else :
        df_user_seq = df_log[val].agg(lambda x: list(x))
    """
    #df_total = pd.DataFrame(df_user_seq).reset_index()
    #df_total.columns = [key,val+"_seq"]
    #df_total[val+"_seq"] = df_total[val+"_seq"].map(lambda x: [str(item) for item in x])

    return df_total

def loadData (path="",num=1000,mode="dev") :
    txdir = path
    print("Step1:Loading Data......")
    if mode == "dev" :
        df_ad = pd.read_csv(txdir + "ad.csv",keep_default_na=False)
        df_click_log = pd.read_csv(txdir + "click_log.csv",nrows=num,keep_default_na=False)
        df_user = pd.read_csv(txdir + "user.csv",keep_default_na=False)
        df_click_log_test = pd.read_csv("./test/click_log.csv",nrows=num,keep_default_na=False)
    else :
        df_ad = pd.read_csv(txdir + "ad.csv",keep_default_na=False)
        df_click_log = pd.read_csv(txdir + "click_log.csv",keep_default_na=False)
        df_user = pd.read_csv(txdir + "user.csv",keep_default_na=False)
        df_click_log_test = pd.read_csv("./test/click_log.csv",keep_default_na=False)
        
    return df_ad,df_click_log,df_user,df_click_log_test

def upload(file_name) :
    secret_id = 'AKIDbWAble2jtshyqIskdJx0nreCZnmaIv1s'  # 替换为用户的secret_id
    secret_key = 'nOkpkk4A8uRntpgOlrgFXZS9oElOmvFu'  # 替换为用户的secret_key
    region = 'ap-shanghai'  # 替换为用户的region
    token = None  # 使用临时密钥需要传入Token，默认为空,可不填

    config = CosConfig(Region=region, Secret_id=secret_id, Secret_key=secret_key, Token=token)  # 获取配置对象
    client = CosS3Client(config)

    '''
    url: 
    https://txad-1252070910.cos.ap-shanghai.myqcloud.com/submission.csv
    '''

    # 文件流 简单上传
    with open(file_name, 'rb') as fp:
        url = client.put_object(
            Bucket='goy-1300206291',  # Bucket由bucketname-appid组成
            Body=fp,
            Key=file_name,
            StorageClass='STANDARD',
            ContentType='text/csv; charset=utf-8'
        )
        print(url)












