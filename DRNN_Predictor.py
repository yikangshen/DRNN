# -*- coding: utf-8 -*-
'''
Created on 2015.4.21

@author: Yikang
'''

import numpy as np

import cPickle

import time, os

from DRNN_logistic import DRNN

if __name__ == '__main__':
    dictfile = open('word_dict_stanford.pkl', 'r')
    word_dict, L = cPickle.load(dictfile)
    dictfile.close()
    nword = len(word_dict.keys())
    
    bmatfile = open('doc_binary_matrix_stanford.pkl', 'r')
    bmats = cPickle.load(bmatfile)
    bmatfile.close()
    
    params_file = open('params_epoch_4_0.00211399209386.pkl','r')
    params = cPickle.load(params_file)
    params_file.close()
    
    sentence_file = open('sentences.txt', 'r')
    sentence_dict = {}
    n = 0
    for line in sentence_file:
        n += 1
        lines = line.strip().split('\t')
        try:
            sentence_dict[n] = lines[1]
        except:
            print line
    sentence_file.close()
    
    rng = np.random.RandomState(1234)
    model = DRNN(rng, L.shape[1], nword, Wf_values=params[0], Wp_values=params[1], L_values=params[2])
    
    n = 0
    print L.shape[0]
    for i in range(L.shape[0]):
        if np.abs(L[i] - params[2][i]).any() > 0:
            n += 1
    print n
    print params[2].max(), params[2].min()
    
    output = open('parse_tree.txt', 'w')
    vecs = []
    n = 0
    for sentence in bmats:
        n += 1
        index0, p0, v0 = model.pv_value(sentence)
        #vecs.append(v0[sentence.shape[0]-1, 0])
        if p0[sentence.shape[0]-1, 0] == 0:
            print n, sentence
        try:
            assert index0.shape[0] == len(sentence_dict[n].split(' '))
        except:
            print sentence_dict[n]
            continue
        try:
            output.write(str(model.get_parse_tree(index0, 0, sentence.shape[0]-1, sentence_dict[n].split(' '))) + '\n')
        except:
            output.write('\n')
            print sentence_dict[n]
    output.close()
    
    vec_file = open('vec_stanford.pkl','w')
    cPickle.dump(vecs, vec_file)
    vec_file.close()