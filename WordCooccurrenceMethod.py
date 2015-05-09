'''
Created on 2015.5.8

@author: Yikang
'''

import gensim.models.word2vec as word2vec
import numpy as np

def cooccurrence(wl1, wl2, co_matrix):
    max_p = 0.0
    sum_p = 0.0
    n = 0
    for w1 in wl1:
        for w2 in wl2:
            if co_matrix[w1,w2] > max_p:
                max_p = co_matrix[w1,w2]
                sum_p += co_matrix[w1,w2]
                n += 1
    return sum_p / n

def parser(co_matrix):
    l = co_matrix.shape[0]
    p_list = []
    index_list = []
    word_list = []
    ind_matrix = np.zeros([l, l], dtype = np.int32)
    p_matrix = np.zeros([l, l], dtype = np.float32)
    for i in range(l):
        p_list.append(1.0)
        word_list.append([i])
        index_list.append((i,i))
        
        ind_matrix[0,i] = i
        p_matrix[0, i] = 1.0
        
    while len(word_list) > 1:
        max_p = 0.0
        max_i = None
        for i in range(len(p_list) - 1):
            new_p = cooccurrence(word_list[i], word_list[i+1], co_matrix)
            if new_p > max_p:
                max_p = new_p
                max_i = i
        max_p =  max_p * p_list[max_i] * p_list[max_i+1]
        middle = index_list[max_i][1]
        left = index_list[max_i][0]
        right = index_list[max_i+1][1]
        max_index = (left, right)
        
        p_list[max_i] = max_p
        del p_list[max_i+1]
        word_list[max_i] = word_list[max_i] + word_list[max_i+1]
        del word_list[max_i+1]
        index_list[max_i] = max_index
        del index_list[max_i+1]
        
        ind_matrix[right-left, left] = middle
        p_matrix[right-left, left] = max_p
        
    return ind_matrix, p_matrix

def get_parse_tree(index_matrix, i, j, sentence = None):
    if i < j:
        k = index_matrix[j-i, i]
        return [
            get_parse_tree(index_matrix, i, k, sentence), 
            get_parse_tree(index_matrix, k+1, j, sentence)
            ]
    elif i == j:
        if sentence != None:
            return sentence[i]
        else:
            return i
    else:
        raise 'error i > j'

if __name__ == '__main__':
    model = word2vec.Word2Vec.load('stanford_word_vector')
    
    sentence_file = open('sentences.txt', 'r')
    sentence_dict = {}
    n = 0
    output = open('parse_tree_cooccurrence.txt','w')
    for line in sentence_file:
        n += 1
        lines = line.strip().split('\t')
        sentence = lines[1].split(' ')
        while ',' in sentence > -1:
            sentence.remove(',')
        l = len(sentence)
        co_matrix = np.zeros([l, l])
        for i in range(l):
            for j in range(l):
                co_matrix[i,j] = model.cooccurrence(sentence[i], sentence[j])
        ind_matrix, p_matrix = parser(co_matrix)
        output.write(str(get_parse_tree(ind_matrix, 0, l-1, sentence)) + '\n')
    output.close()
    sentence_file.close()