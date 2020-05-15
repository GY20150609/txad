import gensim
import pandas as pd


## 输入为序列list

def wvmodel(inputs, epoch=10, outpath=None, params=None):


    print("**********Word2Vector*********")

    if isinstance(inputs, pd.core.series.Series):
        inputs = inputs.tolist()

    num = len(inputs)

    if params is None:
        params = {"min_count":1,  # 保留的最小出现频次
          "alpha":0.1,       # 初始学习率
          "seed":2020,
          "min_alpha":0.001,   # 随着学习进行，学习率线性下降到这个最小数
          "window" : 4,   # 当前和预测单词之间的最大距离
          "size":128,     # 生成词向量的维度
          "compute_loss":True,# 损失(loss)值，如果是True 就会保存
          "workers":8}   # 几个CPU进行跑
        
    min_count = params["min_count"]
    window = params["window"]
    size = params["size"]
    workers = params["workers"]
    alpha = params["alpha"]
    seed = params["seed"]
    compute_loss = params["compute_loss"]
    min_alpha = params["min_alpha"]
    

    print("**********initing*********")
    model = gensim.models.Word2Vec(inputs,
                        min_count=min_count,
                        alpha = alpha,
                        min_alpha = min_alpha,
                        seed = seed,
                            compute_loss = compute_loss,       
                       window=window,
                       size=size,
                       workers=workers)

    print("**********training*********")
    model.train(inputs, total_examples=num, epochs=epoch)

    if outpath:
        print("**********saving*********")
        model.save(outpath)

    return model







