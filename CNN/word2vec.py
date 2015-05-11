# -*- coding: utf-8 -*-
'''
Created on 2015.3.20

@author: Yikang
'''

import gensim.models.word2vec as word2vec
import numpy as np

def cooccurrence(model, w1, w2):
    p1 = 1.0 / (1.0 + np.exp(-np.dot(model.syn0[model.vocab[w1].index], model.syn1neg[model.vocab[w2].index].T)))
    p2 = 1.0 / (1.0 + np.exp(-np.dot(model.syn0[model.vocab[w2].index], model.syn1neg[model.vocab[w1].index].T)))
    
    return (p1 + p2) / 2

if __name__ == '__main__':
    
    sentences = word2vec.LineSentence('data_aclImdb_train_all_training_IMDB.txt')
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=4, workers=4, sg=1, hs=0, negative=10)
    model.save('stanford_word_vector')
    
    #model = word2vec.Word2Vec.load('stanford_word_vector')
    print model.most_similar(['man'])
    print cooccurrence(model, 'he', 'is')
    print cooccurrence(model, 'he', 'are')