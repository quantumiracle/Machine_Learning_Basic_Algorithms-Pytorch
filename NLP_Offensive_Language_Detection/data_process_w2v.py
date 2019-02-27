#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/15 17:36
# @Author  : Kang
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling
from data_process import DataHandle

# from plotly.offline import init_notebook_mode, iplot, plot
# import plotly.graph_objs as go


# def reduce_dimensions(model, plot_in_notebook = False):
#
#     num_dimensions = 2  # final num dimensions (2D, 3D, etc)
#
#     vectors = []        # positions in vector space
#     labels = []         # keep track of words to label our data again later
#     for word in model.wv.vocab:
#         vectors.append(model[word])
#         labels.append(word)
#
#
#     # convert both lists into numpy vectors for reduction
#     vectors = np.asarray(vectors)
#     labels = np.asarray(labels)
#
#     # reduce using t-SNE
#     vectors = np.asarray(vectors)
#     tsne = TSNE(n_components=num_dimensions, random_state=0)
#     vectors = tsne.fit_transform(vectors)
#
#     x_vals = [v[0] for v in vectors]
#     y_vals = [v[1] for v in vectors]
#
#     # Create a trace
#     trace = go.Scatter(
#         x=x_vals,
#         y=y_vals,
#         mode='text',
#         text=labels
#         )
#
#     data = [trace]
#
#     if plot_in_notebook:
#         init_notebook_mode(connected=True)
#         iplot(data, filename='word-embedding-plot')
#     else:
#         plot(data, filename='word-embedding-plot.html')
        

if __name__ == '__main__':

    ex = DataHandle()
    # print(ex.tokenized_corpus)
    # print(ex.vocabulary)
    # print(ex.word2idx)
    sentences = ex.tokenized_corpus
    '''https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/word2vec.ipynb'''
    model = Word2Vec(sentences,min_count=1, window=5, size=20)
    # model.build_vocab(sentences)  # prepare the model vocabulary
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)  # train word vectors
    # plot
    # reduce_dimensions(model)
    print(np.mean(abs(model['comput'])))
    print(model.similarity('woman', 'man'))
    print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
    print(type(model))


