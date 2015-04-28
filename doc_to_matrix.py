'''
Created on 2015.3.14

@author: Yikang
'''

import cPickle
from scipy import sparse
import numpy
import gensim.models.word2vec as word2vec

if __name__ == '__main__':  
    inputtxt = open('stanfordSentimentTreebank\datasetSentences.txt', 'r')
    dictfile = open('word_dict_stanford.pkl', 'r')
    word_dict = cPickle.load(dictfile)
    dictfile.close()
    
    wvmodel = word2vec.Word2Vec.load('stanford_word_vector')
    
    outputtext = open('sentences.txt','w')
    
    nword = len(word_dict.keys())
    bmats = []
    ones = numpy.ones(3000, dtype=numpy.float32)
    n = 0
    max_length = 7
    for line in inputtxt:
        if n == 0:
            n += 1
            continue
        
        lines = line.strip().split('\t')
        sentence = lines[1].split(' ')
        if len(sentence) > max_length: 
            max_length = len(sentence) 
            #continue
        
        outputtext.write(lines[1] + '\n')
        
        mat = []
        for word in sentence:
            try:
                mat.append(wvmodel[word])
            except:
                print word
        bmats.append(numpy.array(mat))
        
        print n
        n += 1
    
    print max_length
    
    outputtext.close()
    
    output = open('doc_binary_matrix_stanford.pkl', 'w')
    cPickle.dump(bmats, output)
    output.close()