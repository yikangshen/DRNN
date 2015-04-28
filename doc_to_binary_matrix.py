# -*- coding: utf-8 -*-
'''
Created on 2015.3.14

@author: Yikang
'''

import cPickle
from scipy import sparse
import numpy

if __name__ == '__main__':  
    dictfile = open('word_dict_stanford.pkl', 'r')
    word_dict, L = cPickle.load(dictfile)
    dictfile.close()
    
    nword = len(word_dict.keys())
    bmats = []
    ones = numpy.ones(300, dtype=numpy.float32)
    n = 0
    max_length = 0
    inputtxt = open('stanford_sentences.txt', 'r')
    outputtext = open('sentences.txt','w')
    for line in inputtxt:
        n += 1
        line = line.lower().strip()
        lines = line.split('\t')
        
        sentence = lines[1].split(' ')
        if len(sentence) > max_length: 
            max_length = len(sentence) 
        
        col = []
        s = lines[0] + '\t'
        for word in sentence:
            if word_dict.has_key(word):
                col.append(word_dict[word])
                s += word + ' '
        s += '\n'
        if len(col) < 2:
            continue
        outputtext.write(s)
        bmats.append(numpy.array(col))
        print n
    print len(bmats)
    inputtxt.close()
    outputtext.close()
    '''
    outputtext = open('sentences_imdb.txt','w')
    inputtxt = open('data_aclImdb_train_all_training_IMDB.txt', 'r')
    for line in inputtxt:
        n += 1
        line = line.lower().strip()
        
        sentence = line.split(' ')
        if len(sentence) > 100: 
            max_length = len(sentence) 
            print line
            continue
        
        col = []
        s = str(n) + '\t'
        for word in sentence:
            if word_dict.has_key(word):
                col.append(word_dict[word])
                s += word + ' '
        s += '\n'
        if len(col) < 2:
            continue
        outputtext.write(s)
        bmats.append(numpy.array(col))
        print n
    inputtxt.close()
    outputtext.close()
    print max_length
    '''
    output = open('doc_binary_matrix_stanford.pkl', 'w')
    cPickle.dump(bmats, output)
    output.close()