# -*- coding: utf-8 -*-
'''
Created on 2015.3.14

@author: Yikang
'''

import cPickle

min_count = 4

import gensim.models.word2vec as word2vec

import numpy

if __name__ == '__main__':
    inputtxt = open('text.txt', 'r')
    vocab = {}
    total_words = 0
    for line in inputtxt:
        line = line.lower()
        line = line.strip()
        sentence = line.split(' ')
        
        for word in sentence:
            total_words += 1
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    inputtxt.close()
    
    textout = open('words_stanford.txt', 'w')
    dict = {}
    n = 0
    L = []
    wvmodel = word2vec.Word2Vec.load('stanford_word_vector')
    for word, v in vocab.items():
        if v >= min_count:
            try:
                L.append(wvmodel[word])
            except:
                print word
                continue
            textout.write(word + '\t' + str(v) + '\n')
            dict[word] = n
            n += 1
    textout.close()
    
    print n
    
    output = open('word_dict_stanford.pkl', 'w')
    cPickle.dump((dict, numpy.array(L, dtype = numpy.float32)), output)
    output.close()
    
            