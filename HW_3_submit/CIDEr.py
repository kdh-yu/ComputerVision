from math import log
from collections import defaultdict, Counter
import numpy as np

def n_gram(sentence, n):
    words = sentence.split()
    ngram = []
    for i in range(len(words)-n+1):
        ngram.append(' '.join(words[i:i+n]))
    return ngram

def TF(sentence, n):
    tf = defaultdict(Counter)
    for s in sentence:
        ngram = n_gram(s, n)
        tf[s].update(ngram)
    return tf

def IDF(sentence, n):
    idf = {}
    df_n = defaultdict(int)
    dnum = len(sentence)
    for s in sentence:
        ngram = set(n_gram(s, n))
        for ng in ngram:
            df_n[ng] += 1
    for ng, cnt in df_n.items():
        idf[ng] = log(dnum/(1+cnt))
    return idf

def TFIDF(tf_, idf_):
    tfidf = defaultdict(dict)
    for sentence, tf_cnt in tf_.items():
        for ng, tf in tf_cnt.items():
            tfidf[sentence][ng] = tf * idf_.get(ng, 0)
    return tfidf

def cos_sim(v1, v2):
    common = list(set(v1.keys()) & set(v2.keys()))
    v1 = np.array([v1[k] for k in common])
    v2 = np.array([v2[k] for k in common])
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    v = np.dot(v1, v2)
    if v1_norm*v2_norm == 0:
        return 0
    else:
        return v/(v1_norm*v2_norm)
    
def CIDEr(caption, reference, N=4):
    vocabulary = reference + caption
    cider_lst = []
    for caps in caption:
        cider = 0
        for n in range(1, N+1):
            tf = TF(vocabulary, n)
            idf = IDF(vocabulary, n)
            tfidf = TFIDF(tf, idf)
            tmp = []
            cap = tfidf[caps]
            for ref in reference:
                ref = tfidf[ref]
                sim = cos_sim(ref, cap)
                tmp.append(sim)
            cider += np.mean(tmp) / n
        cider_lst.append(cider)
    return cider_lst