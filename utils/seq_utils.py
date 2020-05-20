import pandas as pd
from functools import partial
from multiprocessing import Pool
import random
import numpy as np
import collections

wv_indx = 0
iv_indx = 0


def getUserSeq(pairs, key, func):
    return pairs[1][key].agg(func)


def multiProcess(df, func, base="user_id", key="", maxProcess=5):
    """
    Input :
    df : DataFrame include columns [base,key]
    base : column to be grouped
    key : column need to be combined
    maxProcess : max process number

    Output:
    corpus : seq of every distinct base column

    """
    getUserSeqAdd = partial(getUserSeq, key=key, func=func)
    pool = Pool(maxProcess)
    corpus = pool.map(getUserSeqAdd, df.groupby(base, as_index=False))
    pool.close()
    pool.join()
    return corpus


def getDict(words, topK=None):
    """
    Input :
    words : word list
    topK : Fre <= topK will be kept

    Output:
    word2id : word >> index
    id2word : index >> word

    """
    vocabulary = collections.Counter(words)
    count = []
    if topK:
        count.extend(vocabulary.most_common(topK))
    else:
        count.extend(vocabulary.most_common())
    word2id = dict()
    for word, _ in count:
        word2id[word] = len(word2id)
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return vocabulary, word2id, id2word


def generate_single_i2v(sentence, num_skips, word2id, id2word):
    """
    Input :
    sentence : single seq list []
    num_skips : num_skips words will be kept as output words
    word2id : word >> index
    id2word : index >> word

    Output :
    batch_inputs : shape like [num_skips,] or [len(window),]
    batch_labels :shape like [num_skips,1] or [len(window),1]

    single sentence as single batch
    """
    batch_inputs = []
    batch_labels = []
    for i in range(len(sentence)):
        window = list(range(len(sentence)))  # 句子内除该元素外所有元素
        window.remove(i)
        sample_index = random.sample(window, min(num_skips, len(window)))
        input_id = word2id.get(sentence[i])
        for index in sample_index:
            label_id = word2id.get(sentence[index])
            batch_inputs.append(input_id)
            batch_labels.append(label_id)

    batch_inputs = np.array(batch_inputs, dtype=np.int32)
    batch_labels = np.array(batch_labels, dtype=np.int32)
    batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])
    return batch_inputs, batch_labels


def generate_single_w2v(sentence, num_skips, skip_window, word2id, id2word):
    """
    Input :
    sentence : single seq list []
    num_skips : num_skips words will be kept as output words
    skip_window : window size current word to expand
    word2id : word >> index
    id2word : index >> word

    Output :
    batch_inputs : shape like [num_skips,] or [len(window),]
    batch_labels :shape like [num_skips,1] or [len(window),1]

    single sentence as single batch
    """
    batch_inputs = []
    batch_labels = []
    sen_len = len(sentence)
    for i in range(sen_len):
        start = i - skip_window if i - skip_window > 0 else 0
        end = i + skip_window if i + skip_window < sen_len else sen_len - 1
        new_skip_num = end - start if num_skips > (end - start) else num_skips
        window = list(range(start, end + 1))
        window.remove(i)
        sample_index = random.sample(window, new_skip_num)
        input_id = word2id.get(sentence[i])
        for index in sample_index:
            label_id = word2id.get(sentence[index])
            batch_inputs.append(input_id)
            batch_labels.append(label_id)
    batch_inputs = np.array(batch_inputs, dtype=np.int32)
    batch_labels = np.array(batch_labels, dtype=np.int32)
    batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])
    return batch_inputs, batch_labels


def generate_batch_w2v(sentences, num_skips, skip_window, word2id, id2word, sen_size=16):
    global wv_indx
    sen_num = len(sentences)
    batch_inputs = []
    batch_labels = []
    for i in range(sen_size):
        wv_indx = wv_indx % sen_num
        b_inputs, b_labels = generate_single_w2v(sentences[wv_indx], num_skips, skip_window, word2id, id2word)
        batch_inputs.append(b_inputs)
        batch_labels.append(b_labels)
        wv_indx = (wv_indx + 1) % sen_num
    batch_inputs = np.concatenate(batch_inputs).astype(np.int32)
    batch_labels = np.concatenate(batch_labels).astype(np.int32)
    batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])
    return batch_inputs, batch_labels


def generate_batch_i2v(sentences, num_skips, word2id, id2word, sen_size=16):
    global iv_indx
    sen_num = len(sentences)
    batch_inputs = []
    batch_labels = []
    for i in range(sen_size):
        iv_indx = iv_indx % sen_num
        b_inputs, b_labels = generate_single_i2v(sentences[iv_indx], num_skips, word2id, id2word)
        batch_inputs.append(b_inputs)
        batch_labels.append(b_labels)
        iv_indx = (iv_indx + 1) % sen_num
    batch_inputs = np.concatenate(batch_inputs).astype(np.int32)
    batch_labels = np.concatenate(batch_labels).astype(np.int32)
    batch_labels = np.reshape(batch_labels, [batch_labels.__len__(), 1])
    return batch_inputs, batch_labels




