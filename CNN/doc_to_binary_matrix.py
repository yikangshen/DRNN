# -*- coding: utf-8 -*-
'''
Created on 2015.3.14

@author: Yikang
'''

import cPickle
import numpy
import random

import gensim.models.word2vec as word2vec

if __name__ == '__main__':  
    model = word2vec.Word2Vec.load('stanford_word_vector')
    nword = len(model.vocab.keys())
    x_train = []
    y_train = []
    ones = numpy.ones(300, dtype=numpy.float32)
    n = 0
    max_length = 0
    inputtxt = open('data_aclImdb_train_neg_training_IMDB.txt', 'r')
    for line in inputtxt:
        n += 1
        line = line.lower().strip()
        sentence = line.split(' ')
        if len(sentence) > max_length: 
            max_length = len(sentence) 
        
        col = []
        for word in sentence:
            if model.vocab.has_key(word):
                col.append(model.vocab[word].index)
        if len(col) < 2:
            continue
        x_train.append(numpy.array(col))
        y_train.append(0)
        print n
    print len(x_train)
    inputtxt.close()
    
    inputtxt = open('data_aclImdb_train_pos_training_IMDB.txt', 'r')
    for line in inputtxt:
        n += 1
        line = line.lower().strip()
        sentence = line.split(' ')
        if len(sentence) > max_length: 
            max_length = len(sentence) 
        
        col = []
        for word in sentence:
            if model.vocab.has_key(word):
                col.append(model.vocab[word].index)
        if len(col) < 2:
            continue
        x_train.append(numpy.array(col))
        y_train.append(1)
        print n
    print len(x_train)
    inputtxt.close()
    
    t = zip(x_train, y_train)
    random.shuffle(t)
    x_train, y_train = zip(*t)
    
    x_test = []
    y_test = []
    inputtxt = open('data_aclImdb_test_neg_training_IMDB.txt', 'r')
    for line in inputtxt:
        n += 1
        line = line.lower().strip()
        sentence = line.split(' ')
        if len(sentence) > max_length: 
            max_length = len(sentence) 
        
        col = []
        for word in sentence:
            if model.vocab.has_key(word):
                col.append(model.vocab[word].index)
        if len(col) < 2:
            continue
        x_test.append(numpy.array(col))
        y_test.append(0)
        print n
    print len(x_test)
    inputtxt.close()
    
    inputtxt = open('data_aclImdb_test_pos_training_IMDB.txt', 'r')
    for line in inputtxt:
        n += 1
        line = line.lower().strip()
        sentence = line.split(' ')
        if len(sentence) > max_length: 
            max_length = len(sentence) 
        
        col = []
        for word in sentence:
            if model.vocab.has_key(word):
                col.append(model.vocab[word].index)
        if len(col) < 2:
            continue
        x_test.append(numpy.array(col))
        y_test.append(1)
        print n
    print len(x_test)
    inputtxt.close()

    output = open('doc_binary_matrix_stanford.pkl', 'w')
    cPickle.dump(((x_train,y_train), (x_test,y_test)), output)
    output.close()