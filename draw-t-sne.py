#!/usr/bin/python
# encoding=utf-8
# -*- coding:utf-8 -* 
#author: fanzhiyun
#date：20200613



# 切换工作路径
import os
import sys
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] )

import numpy as np
from numpy import *
import numpy as np

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class chj_data(object):
    def __init__(self,data,target):
        self.data=data
        self.target=target


def load_lang(lexicon):
    lang2int = {}
    with open(lexicon, 'r') as fo:
        lines = fo.readlines()
        for i, line in enumerate(lines):
            lang = line.strip().split()[0]
            lang2int[lang] = i
    return lang2int


def chj_load_file(fdata, lang2int):
    utt2target = {}
    feature = []
    target = []
    with open(fdata, 'r') as fo:
        for line in fo:
            uttid = line.strip().split()[0]
            lang = line.strip().split()[1]
            embedding = line.strip().split()[1:]
            feature.append(embedding)
            target.append(int(lang))
    feature=np.array(feature)
    target = np.array(target)
    res=chj_data(feature,target)
    return res

def guass_random():
    a=np.random.multivariate_normal(mean=[0,0,0,0], cov=np.identity(4), size=100)
    a_l=np.array(['#0000FF' for i in range(100)] + ['#FFF5EE' for i in range(100)])
    b=np.random.multivariate_normal(mean=[1,1,1,1], cov=np.identity(4), size=100)
    b_l=np.array(['#FFF5EE' for i in range(100)])
    c=np.concatenate((a,b),axis=0)
    return c,a_l


if __name__ == "__main__":


    #embedding="./exp/wav2vev2_small_finetune_AP18_train20s/results/test_all_embedding_freeze"
    #embedding="./exp/wav2vev2_small_finetune_VOX1_bak/results/test_embedding.spk0-9"
    embedding="./exp/wav2vev2_small_finetune_VOX1_for_freeze/results/test_embedding_freeze.spk0-9"
    lexicon="../fairseq/exp/data/AP18/dict.ltr.txt"
    lang2int = load_lang(lexicon)
    #iris = load_iris() # 使用sklearn自带的测试文件
    iris = chj_load_file(embedding, lang2int)
    #rand,label = guass_random()
    #X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(iris.data[:100])
    #X_tsne = TSNE(n_components=2,init='pca',perplexity=50,learning_rate=100).fit_transform(iris.data) #draw on the whole test_all 
    X_tsne = TSNE(n_components=2,init='pca',learning_rate=100).fit_transform(iris.data)
    #X_pca = PCA().fit_transform(iris.data)
    fig = plt.figure()
    plt.axis('off')  #去掉坐标轴
    #plt.figure(figsize=(12, 6))
    #plt.subplot(121)
    #plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target[:100])
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
    #plt.subplot(122)
    #plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
    #plt.colorbar()
    #plt.show() 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0.03,0.03)
    fig.savefig('sid-vox1-freeze-test-spk0-9.eps', dpi=600, format='eps')
    
