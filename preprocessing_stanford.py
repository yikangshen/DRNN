# -*- coding: utf-8 -*-
'''
Created on 2015.3.14

@author: Yikang
'''

import cPickle
from scipy import sparse
import numpy

if __name__ == '__main__':  
    inputtxt = open('data\stanfordSentimentTreebank\datasetSentences.txt', 'r')
    
    outputtext = open('stanford_sentences.txt','w')
    
    n = 0
    for line in inputtxt:
        line = line.lower()
        if n == 0:
            n += 1
            continue
        
        while line.find('-lrb-') > -1:
            line = line.replace('-lrb-', '(')
            line = line.replace('-rrb-', ')')
        while line.find('-') > -1:
            line = line.replace('-', ' ')
        while line.find('  ') > -1:
            line = line.replace('  ', ' ')
        
        #lines = line.split('\t')
        outputtext.write(line)
        
        print n
        n += 1
    
    outputtext.close()