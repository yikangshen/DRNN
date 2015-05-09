# -*- coding: utf-8 -*-
'''
Created on 2015.3.20

@author: Yikang
'''

import gensim.models.word2vec as word2vec

if __name__ == '__main__':
    sentences = word2vec.LineSentence('text.txt')
    model = word2vec.Word2Vec(sentences, size=100, window=5, min_count=4, workers=4, sg=1, hs=0, negative=10)
    model.save('stanford_word_vector')
    
    print model.most_similar(['man'])
    print model.cooccurrence('he', 'is')
    print model.cooccurrence('he', 'are')