#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/15 17:36
# @Author  : Kang
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm
import pandas as pd
from gensim import corpora
from gensim import models
from data_process import DataHandle



if __name__ == '__main__':

    ex = DataHandle()
    print(ex.tokenized_corpus)
    processed_corpus=ex.tokenized_corpus
    # print(ex.vocabulary)
    # print(ex.word2idx)
    # print(ex.idx2word)
    dictionary = corpora.Dictionary(processed_corpus)
    # bag-of-words
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    # print(bow_corpus)

    # train the model
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    # different models
    ''' https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Topics_and_Transformations.ipynb'''
    model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    # model = models.RpModel(corpus_tfidf, num_topics=500)
    # model = models.LdaModel(bow_corpus, id2word=dictionary, num_topics=100)
    # model = models.HdpModel(bow_corpus, id2word=dictionary)

    # transform the "system minors" string
    # print(model[dictionary.doc2bow("minors".lower().split())])
    # print(model[dictionary.doc2bow("minor".lower().split())])
    # print(model[dictionary.doc2bow("m#d,s".lower().split())])


