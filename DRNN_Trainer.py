# -*- coding: utf-8 -*-
'''
Created on 2015.4.12

@author: Yikang
'''

import numpy as np

import cPickle

import time, os

os.environ['MKL_NUM_THREADS'] = '1'

from DRNN_kernel import DRNN

max_length = 60

fixL = False

from multiprocessing import Pool, cpu_count

def pv(args):
    model, sentences = args
    gps = []
    for param in model.params:
        gps.append(np.zeros_like(param, dtype=np.float32))
    dpp = 0.0
    for n, sentence, random_row in sentences:
        index0, p0, v0 = model.pv_value(sentence)
        index1, p1, v1 = model.pv_value(random_row)
        pp0 = p0[sentence.shape[0]-1,0]
        pp1 = p1[sentence.shape[0]-1,0]
        dpp += pp0 - pp1
        if pp0 - pp1 > 0.1:
            #cost = 0
            gparams_value = [0, 0, 0, 0]
        else:
            if pp0 > 0:
                dp0 = model.backprop(0, sentence.shape[0]-1, -1.0 * (0.1 - (pp0 - pp1)), np.zeros(model.size, dtype=np.float32), index0, p0, v0)
            else:
                print n, sentence
                continue
            if pp1 > 0:
                dp1 = model.backprop(0, sentence.shape[0]-1, 1.0 * (0.1 - (pp0 - pp1)), np.zeros(model.size, dtype=np.float32), index1, p1, v1)
            else:
                print n, sentence
                continue
            gparams_value = [d0 + d1 for d0,d1 in zip(dp0, dp1)]
            #gparams_value = [dWf0 + dWf1, dWp0 + dWp1, dbp0 + dbp1, dL0 + dL1]
        for i in range(len(gps)):
            gps[i] = gps[i] + gparams_value[i]
      
        #index_list0 = model.get_parse_tree(index0, 0, sentence.shape[0]-1)
        #index_list1 = model.get_parse_tree(index1, 0, random_row.shape[0]-1)
        print n, 'p:', pp0, ',', pp1, 'p0-p1:', pp0 - pp1, 'p0/p2:', pp0 / pp1
        #print index_list0
        #print index_list1
       
    return (dpp/len(sentences), gps)

def train_DRNN(nword, bmats, max_length, params=None, L = None,
            margin = 0.1, learning_rate=0.01, 
            L1_reg=0.00, L2_reg=0.0001, 
            n_epochs=100, batch_size=200):
    rng = np.random.RandomState(1234)
        
    if params != None:
        model = DRNN(rng, params[2].shape[1], nword, Wf_values=params[0], Wp_values=params[1], L_values=params[2])
    elif L != None:
        model = DRNN(rng, L.shape[1], nword, L_values=L)
    else:
        model = DRNN(rng, 100, nword)
    
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    n_train_batches = len(bmats) / batch_size
    n_per_process = batch_size / cpu_count()
    
    pool = Pool()
    
    epoch = 0
    average_diff = []
    logfille = open('training_log.txt', 'w')
    logfille.write(str(time.clock())+'\n')
    logfille.flush()
    while epoch < n_epochs:
        epoch = epoch + 1
        differencelist = []
        n = 0
        for minibatch_index in xrange(n_train_batches):
            process_task = []
            count = 0
            for sentence in bmats[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]:
                n += 1
                
                if count == 0:
                    sentences = []
                    
                l = sentence.shape[0]
                if l < 2:
                    continue
                t = sentence[0:l-1].copy()
                np.random.shuffle(t)
                random_row = sentence.copy()
                random_row[0:l-1] = t
                '''
                if np.random.randint(2) > 0:
                    #change one word
                    random_row[np.random.randint(l)] = np.random.randint(nword)
                else:
                    #change two word position
                    i1 = np.random.randint(l)
                    i2 = np.random.randint(l)
                    while i1 == i2:
                        i2 = np.random.randint(l)
                    t = random_row[i1]
                    random_row[i1] = random_row[i2]
                    random_row[i2] = t
                '''
                sentences.append((n, sentence, random_row))
                count += 1
                
                if count > n_per_process:
                    process_task.append((model, sentences))
                    count = 0
            process_task.append((model, sentences))
                    
            results = pool.map(pv, process_task)
            
            gps = []
            for param in model.params:
                gps.append(np.zeros_like(param, dtype=np.float32))
            for dpp, gparams_value in results:
                for i in range(len(gps)):
                    gps[i] = gps[i] + gparams_value[i]            
                differencelist.append(dpp)
                
            if fixL:
                for i in range(len(gps)-1):
                    model.params[i] -= learning_rate * gps[i] / batch_size
            else:
                for i in range(len(gps)):
                    model.params[i] -= learning_rate * gps[i] / batch_size
        
        average_diff.append(np.mean(differencelist))
        
        output = open('params_epoch_'+str(epoch)+'_'+str(np.mean(differencelist))+'.pkl', 'w')
        cPickle.dump(model.params, output)
        output.close()
            
        logfille.write(str(time.clock())+'\n')
        logfille.write('epoch:' + str(epoch) + 'p0, p1 difference:' + str(np.mean(differencelist)) + '\n')
        logfille.flush()
    logfille.close()

if __name__ == '__main__':
    dictfile = open('word_dict_stanford.pkl', 'r')
    word_dict, L = cPickle.load(dictfile)
    dictfile.close()
    nword = len(word_dict.keys())
    
    bmatfile = open('doc_binary_matrix_stanford.pkl', 'r')
    bmats = cPickle.load(bmatfile)
    bmatfile.close()
    
    '''
    params_file = open('params_epoch_6_0.00160522288763.pkl','r')
    params = cPickle.load(params_file)
    params_file.close()
    '''
    train_DRNN(nword, bmats, max_length, L=L.astype(np.float32))
