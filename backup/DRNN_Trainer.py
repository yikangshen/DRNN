'''
Created on 2015.4.12

@author: Yikang
'''

import numpy as np

import theano
import theano.tensor as T

import cPickle

import time, os

from backup import DRNN9

max_length = 60

#os.environ['MKL_NUM_THREADS'] = '8'

def train_DRNN(nword, bmats, max_length, L = None,
            margin = 0.1, learning_rate=0.01, 
            L1_reg=0.00, L2_reg=0.0001, 
            n_epochs=10, batch_size=40):
    rng = np.random.RandomState(1234)
    
    if L == None:
        model = DRNN9(rng, 100, nword, max_length)
    else:
        model = DRNN9(rng, L.shape[1], nword, max_length, L_values=L)
    
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    
    vec = T.ivector('vec')
    #start = T.iscalar('start')
    l = T.iscalar('l')
    pf_vec0 = T.fscalar('pf_vec0')
    pf_vec1 = T.fscalar('pf_vec1')
    
    print 'model...'
    cost = (
            -(pf_vec0 - pf_vec1 - margin)
            #+ L1_reg * model.L1
            #+ L2_reg * model.L2_sqr
            )
    gparams = T.grad(cost, [pf_vec0, pf_vec1])
    indexf, pf, vf = model.pv_function(vec)
                
    #compile_mode = theano.Mode(linker='cvm', optimizer='fast_compile')
    
    print 'function...'
    gparams_function = theano.function(
                                    inputs=[pf_vec0, pf_vec1],
                                    outputs=gparams,
                                    )
    
    ipv_function = theano.function(
                                inputs=[vec],
                                outputs=[indexf, pf, vf],
                                )

    n_train_batches = len(bmats) / batch_size
    
    epoch = 0
    average_diff = []
    while epoch < n_epochs:
        epoch = epoch + 1
        differencelist = []
        n = 0
        for minibatch_index in xrange(n_train_batches):
            gps = None
            for sentence in bmats[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]:
                n += 1
                
                random_row = np.random.randint(0, nword, sentence.shape[0])
                
                [index0, p0, v0] = ipv_function(sentence)
                #p0 = p0 * (2**(sentence.shape[0]-1))
                index_list = model.get_parse_tree(index0, 0, sentence.shape[0]-1)
                [index1, p1, v1] = ipv_function(random_row)
                #p1 = p1 * (2**(sentence.shape[0]-1))
                pp0 = p0[sentence.shape[0]-1,0]
                pp1 = p1[sentence.shape[0]-1,0]
                
                if pp0 - pp1 > margin:
                    #cost = 0
                    gparams_value = [0, 0]
                else:
                    [dp0, dp1] = gparams_function(pp0, pp1)
                    dWp0, dWf0 = model.backprop(0, sentence.shape[0]-1, dp0, np.zeros(model.size, dtype=np.float32), index0, p0, v0)
                    dWp1, dWf1 = model.backprop(0, sentence.shape[0]-1, dp1, np.zeros(model.size, dtype=np.float32), index1, p1, v1)
                    gparams_value = [dWf0 + dWf1, dWp0 + dWp1]
                    
                if gps == None:
                    gps = gparams_value
                else:
                    for i in range(len(gps)):
                        gps[i] = gps[i] + gparams_value[i]
                        
                print n, 'p:', pp0, ',', pp1, 'p0-p1:', pp0 - pp1, 'p0/p2:', pp0 / pp1
                print index_list
                differencelist.append(pp0 - pp1)
                
            for i in range(len(gps)):
                model.params[i].set_value(model.params[i].get_value() - learning_rate * gps[i] / batch_size)
        
        print 'epoch:', epoch, 'p0, p1 difference:', np.mean(differencelist)
        average_diff.append(np.mean(differencelist))
        
        print average_diff
        
        params = []
        for param in model.params:
            params.append(param.get_value())
        output = open('params_epoch_'+str(epoch)+'_'+str(np.mean(differencelist))+'.pkl', 'w')
        cPickle.dump(params, output)
        output.close()
            
        print 'epoch:', epoch, 'p0, p1 difference:', np.mean(differencelist)

if __name__ == '__main__':
    dictfile = open('word_dict_stanford.pkl', 'r')
    word_dict, L = cPickle.load(dictfile)
    dictfile.close()
    nword = len(word_dict.keys())
    
    bmatfile = open('doc_binary_matrix_stanford.pkl', 'r')
    bmats = cPickle.load(bmatfile)
    bmatfile.close()
    
    train_DRNN(nword, bmats, max_length, L=L)
