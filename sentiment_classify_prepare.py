# -*- coding: utf-8 -*-
'''
Created on 2015.4.21

@author: Yikang
'''

import numpy as np

import cPickle
import random

if __name__ == '__main__':
    sentiment_file = open('data\stanfordSentimentTreebank\sentiment_labels.txt', 'r')
    sentiment_label = {}
    n = 0
    for line in sentiment_file:
        lines = line.strip().split('|')
        if n > 0:
            sentiment_label[int(lines[0])] = float(lines[1])
        n += 1
    sentiment_file.close()
        
    dict_file = open('data\stanfordSentimentTreebank\dictionary.txt', 'r')
    phrase_dict = {}
    for line in dict_file:
        line = line.decode('utf-8')
        lines =line.strip().split('|')
        phrase_dict[lines[0]] = int(lines[1])
    dict_file.close()
    
    sentence_file = open('data\stanfordSentimentTreebank\datasetSentences.txt', 'r')
    sentence_dict = {}
    n = 0
    for line in sentence_file:
        line = line.decode('utf-8')
        line = line.replace('-LRB-', '(')
        line = line.replace('-RRB-', ')')
        lines = line.strip().split('\t')
        if n > 0:
            sentence_dict[int(lines[0])] = lines[1]
        n += 1
    sentence_file.close()
    
    datasplit_file = open('data\stanfordSentimentTreebank\datasetSplit.txt', 'r')
    split_dict = {}
    n = 0
    for line in datasplit_file:
        lines = line.strip().split(',')
        if n > 0:
            split_dict[int(lines[0])] = int(lines[1])
        n += 1
    datasplit_file.close()
    
    vec_file = open('vec_stanford.pkl','r')
    vecs = cPickle.load(vec_file)
    vec_file.close()
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_valid, y_valid = [], []
    
    n0 = 0
    n1 = 1
    for i in range(len(vecs)):
        #print sentence_dict[i+1].encode('utf-8')
        try:
            senti = sentiment_label[phrase_dict[sentence_dict[i+1]]]
        except:
            print sentence_dict[i+1]
            continue
        #print vecs[i]
        print senti, sentence_dict[i+1]
        if (senti > 0.4) and (senti <= 0.6):
            continue
        if senti > 0.6:
            senti = 1
            n1 += 1
        if senti <= 0.4:
            senti = 0
            n0 += 1
        if split_dict[i+1] == 1:
            x_train.append(vecs[i])
            y_train.append(senti)
        elif split_dict[i+1] == 2:
            x_test.append(vecs[i])
            y_test.append(senti)
        else:
            x_valid.append(vecs[i])
            y_valid.append(senti)
    
    print len(x_train), len(x_valid), len(x_test)
    t = zip(x_train, y_train)
    random.shuffle(t)
    x_train, y_train = zip(*t)
    
    sentiment_trainingdata = open('sentiment_trainingdata.pkl', 'w')
    cPickle.dump(((x_train,y_train), (x_valid, y_valid), (x_test, y_test)), sentiment_trainingdata)
    sentiment_trainingdata.close()
    
    print y_train