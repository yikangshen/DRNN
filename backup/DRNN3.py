'''
2015.3.12
by hmwv1114
'''

import numpy as np
import scipy

import theano
import theano.tensor as T

from theano import sparse

import cPickle

import time
import sys, os

sys.setrecursionlimit(10000)

max_length = 10

class DRNN(object):
	def __init__(self, rng, size, N_word, max_length, Wf=None, Wp=None, L=None, activation = T.tanh):
		self.size = size
		self.max_length = max_length
		
		#initial Wf, bf
		if Wf is None:
			Wf_values = np.asarray(
								rng.uniform(
										low=-np.sqrt(6. / (size + size*2)),
										high=np.sqrt(6. / (size + size*2)),
										size=(size, size*2+1)
										),
								dtype=theano.config.floatX
								)
			if activation == theano.tensor.nnet.sigmoid:
				Wf_values *= 4

			Wf = theano.shared(value=Wf_values, name='Wf', borrow=True)

		self.Wf = Wf
		
		#initial Wp, bp
		if Wp is None:
			Wp_values = np.asarray(
								rng.uniform(
										low=-np.sqrt(6. / (size*2)),
										high=np.sqrt(6. / (size*2)),
										size=(2*size+1,)
										),
								dtype=theano.config.floatX
								)
			Wp = theano.shared(value=Wp_values, name='Wp', borrow=True)
		self.Wp = Wp
		if L is None:
			L_values = np.asarray(
					rng.uniform(
					low=-np.sqrt(6. / (N_word)),
                    high=np.sqrt(6. / (N_word)),
                    size=(N_word, size)
                ),
                dtype=theano.config.floatX
            )

			L = theano.shared(value=L_values, name='L', borrow=True)
			
			self.L = L
			
			self.params = [
						self.Wf,
						self.Wp,
						self.L
						]
		else:
			self.L = theano.shared(L, name='L', borrow=True)
		
			self.params = [
						self.Wf,
						self.Wp, 
						#self.L
						]
		
		self.L1 = (
            abs(self.Wf).sum()
            + abs(self.Wp).sum()
        )
		
		self.L2_sqr = (
            (self.Wf ** 2).sum()
            + (self.Wp ** 2).sum()
        )
		

	def f_function(self, v1, v2):
		return T.tanh(T.dot(self.Wf, T.concatenate([v1, v2, [np.float32(1.0)]])))
	
	def g_function(self, v1, v2):
		return T.nnet.sigmoid(T.dot(self.Wp, T.concatenate([v1, v2, [np.float32(1.0)]])))
	
	#p_scalar_function, v_vector_function = pv_function(sentence_matrix)
	def pv_function(self, tensor_input):
		indexf_matrix = theano.shared(
									np.ones(
										(self.max_length, self.max_length), 
										dtype=theano.config.floatX
										),
									name = 'indexf_matrix',
									borrow=True
									)
		pf_matrix = theano.shared(
								np.eye(
										self.max_length, 
										dtype=theano.config.floatX
										),
								name = 'pf_matrix',
								borrow=True
								)
		vf_matrix = theano.shared(
								np.zeros(
										(self.max_length, self.max_length, self.size), 
										dtype=theano.config.floatX
										),
								name = 'vf_matrix',
								borrow=True
								)

		results, updates = theano.reduce(
				fn = lambda i, t_vf_matrix, L, t_tensor_input: T.set_subtensor(t_vf_matrix[i, i], L[t_tensor_input[i]]),
				outputs_info = vf_matrix,
				sequences=[T.arange(tensor_input.shape[0])],
				non_sequences=[self.L, tensor_input],
				name = 'vf_matrix prepare'
				)
		vf_matrix = results
		
		for i in range(1,self.max_length):
			'''
			for j in range(self.max_length-i):
				new_index, new_pf, new_vf = self._pv_function1(j, j+i, pf_matrix, vf_matrix)
				indexf_matrix = T.set_subtensor(indexf_matrix[j, j+i], new_index)
				pf_matrix = T.set_subtensor(pf_matrix[j, j+i], new_pf)
				vf_matrix = T.set_subtensor(vf_matrix[j, j+i], new_vf)
			'''
			results, updates = theano.map(
				fn = lambda j, pf_matrix, vf_matrix, i: self._pv_function1(j, j+i, pf_matrix, vf_matrix),
				sequences=[T.arange(self.max_length-i)],
				non_sequences = [pf_matrix, vf_matrix, i],
				#name = 'pv function'
				)
			for j in range(self.max_length-i):
				indexf_matrix = T.set_subtensor(indexf_matrix[j, j+i], results[0][j])
				pf_matrix = T.set_subtensor(pf_matrix[j, j+i], results[1][j])
				vf_matrix = T.set_subtensor(vf_matrix[j, j+i], results[2][j])
			
		return indexf_matrix, pf_matrix, vf_matrix
	
	def _pv_function0(self, tensor_left, tensor_right, pf_matrix, vf_matrix):
		pf = theano.shared(np.zeros(tensor_right - tensor_left, dtype=np.float32))
		#vf = theano.shared(np.zeros((tensor_right - tensor_left, self.size), dtype=np.float32))
		for i in range(tensor_left, tensor_right):
			new_pf = self.get_new_p(tensor_left, tensor_right, i, pf_matrix, vf_matrix)
			
			pf = T.set_subtensor(pf[i-tensor_left], new_pf)
			#vf = T.set_subtensor(vf[i-tensor_left], new_vf)
		max_pf, index = T.max_and_argmax(pf)
		return index + tensor_left, max_pf, self.get_new_v(tensor_left, tensor_right, index + tensor_left, vf_matrix)
	
	def get_new_pv(self, tensor_left, tensor_right, i, pf_matrix, vf_matrix):
		p1f = pf_matrix[tensor_left, i]
		v1f = vf_matrix[tensor_left, i]
		p2f = pf_matrix[i+1, tensor_right]
		v2f = vf_matrix[i+1, tensor_right]
		
		new_pf = self.g_function(v1f,v2f) * p1f * p2f
		new_vf = self.f_function(v1f,v2f)
		
		return [new_pf, new_vf]
	
	def _pv_function1(self, tensor_left, tensor_right, pf_matrix, vf_matrix):
		results, updates = theano.map(
				fn = lambda i, tensor_left, tensor_right, pf_matrix, vf_matrix: self.get_new_p(tensor_left, tensor_right, i, pf_matrix, vf_matrix),
				sequences=[T.arange(tensor_left, tensor_right)],
				non_sequences = [tensor_left, tensor_right, pf_matrix, vf_matrix],
				name = 'pv function'
				)
		max_pf, index = T.max_and_argmax(results, axis=0)
		return [index + tensor_left, max_pf, self.get_new_v(tensor_left, tensor_right, index + tensor_left, vf_matrix)]
	
	def get_new_p(self, tensor_left, tensor_right, i, pf_matrix, vf_matrix):
		p1f = pf_matrix[tensor_left, i]
		v1f = vf_matrix[tensor_left, i]
		p2f = pf_matrix[i+1, tensor_right]
		v2f = vf_matrix[i+1, tensor_right]
		
		new_pf = self.g_function(v1f,v2f) * p1f * p2f
		#new_vf = self.f_function(v1f,v2f)
		
		return new_pf
	
	def get_new_v(self, tensor_left, tensor_right, i, vf_matrix):
		v1f = vf_matrix[tensor_left, i]
		v2f = vf_matrix[i+1, tensor_right]
		
		new_vf = self.f_function(v1f,v2f)
		
		return new_vf
	
def get_index(index_matrix, i, j):
	if i < j:
		k = index_matrix[i][j]
		return [
			k.tolist(), 
			get_index(index_matrix, i, k), 
			get_index(index_matrix, k+1, j)
			]
	elif i == j:
		return [i]
	else:
		raise 'error i > j'
				
def train_DRNN(nword, bmats, max_length, L = None,
			margin = 0.1, learning_rate=0.01, 
			L1_reg=0.00, L2_reg=0.0001, 
			n_epochs=3, batch_size=20):
	rng = np.random.RandomState(1234)
	
	if L == None:
		model = DRNN(rng, 100, nword, max_length)
	else:
		model = DRNN(rng, L.shape[1], nword, max_length, L=L)
	
	best_iter = 0
	test_score = 0.
	start_time = time.clock()
	
	ones = np.ones(3000, dtype = np.float32)
	mat0 = sparse.matrix('csr', 'mat0')
	mat1 = sparse.matrix('csr', 'mat1')
	#start = T.iscalar('start')
	end = T.iscalar('end')
	
	print 'functions...'
	indexf0, pf0, vf0 = model.pv_function(mat0)
	indexf1, pf1, vf1 = model.pv_function(mat1)
	p_difference = pf0[0,end] - pf1[0,end]
	cost = (
		T.max([0, -(p_difference - margin)])
		+ L1_reg * model.L1
        + L2_reg * model.L2_sqr
        )
	
	print 'gparams...'
	gparams = [
			T.grad(cost, param) 
			for param in model.params
			]
	updates = [
			(param, param - learning_rate * gparam)
			for param, gparam in zip(model.params, gparams)
			]
				
	compile_mode = theano.Mode(linker='cvm', optimizer='fast_compile')
	
	print 'p_difference_model...'
	p_difference_model = theano.function(
								inputs=[mat0, mat1, end],
			    				outputs=p_difference,
			        			#mode=compile_mode,
			        			on_unused_input='warn'
			        			)
	
	print 'train_model...'
	gparams_model = theano.function(
								inputs=[mat0, mat1, end],
			    				outputs=gparams,
			        			mode=compile_mode,
			        			on_unused_input='warn'
			        			)
	
	print 'index_model...'
	index_model = theano.function(
								inputs=[mat0],
			    				outputs=indexf0,
			        			#mode=compile_mode,
			        			on_unused_input='warn'
			        			)
	
	n_train_batches = len(bmats) / batch_size
	
	epoch = 0
	while epoch < n_epochs:
		epoch = epoch + 1
		differencelist = []
		n = 0
		for minibatch_index in xrange(n_train_batches):
			gps = None
			for sentence in bmats[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]:
				n += 1
				
				random_row = np.random.randint(0, nword, sentence.shape[0])
				random_mat = scipy.sparse.csr_matrix(
														(
															ones[:sentence.shape[0]], 
															(range(sentence.shape[0]), random_row)
														), 
														shape=[sentence.shape[0], nword]
													)
				
				gparams_value = gparams_model(sentence, random_mat, sentence.shape[0]-1)
				
				if gps == None:
					gps = gparams_value
				else:
					for i in range(len(gps)):
						gps[i] = gps[i] + gparams_value[i]
						
				p_difference_value = p_difference_model(sentence, random_mat, sentence.shape[0]-1)
				print n, p_difference_value
				print get_index(index_model(sentence.toarray()), 0, sentence.shape[0]-1)
				differencelist.append(p_difference_value)
				
			for i in range(len(gps)):
				model.params[i].set_value(model.params[i].get_value() - learning_rate * gps[i] / batch_size)
			
		print 'epoch:', epoch, 'p0, p1 difference:', np.mean(differencelist)
		#print model.L.get_value()

if __name__ == '__main__':
	dictfile = open('word_dict_stanford.pkl', 'r')
	word_dict, L = cPickle.load(dictfile)
	dictfile.close()
	nword = len(word_dict.keys())
	'''
	bmatfile = open('doc_binary_matrix_stanford.pkl', 'r')
	bmats = cPickle.load(bmatfile)
	bmatfile.close()
	
	train_DRNN(nword, bmats, max_length, L=L)
	'''
	length = 20	
	rng = np.random.RandomState(1234)
	model = DRNN(rng, L.shape[1], nword, length, L=L)
	
	#start
	print 'start'
	start_time = time.clock()
	vec0 = T.ivector('vec0')
	end = T.iscalar('end')
	
	#time1
	print 'model'
	time1 = time.clock()	
	indexf, pf, vf = model.pv_function(vec0)
	
	#time2
	print 'functon'
	time2 = time.clock()
	#indexfunction = pfunction = theano.function(
	#						inputs=[mat0],
	#						outputs=indexf,
	#						#mode='FAST_COMPILE',
	#						)
	pfunction = theano.function(
							inputs=[vec0],
							outputs=[pf, indexf],
							#mode=theano.Mode(linker='cvm', optimizer='fast_compile'),
							)
	#vfunction = theano.function(
	#					inputs=[mat0],
	#					outputs=vf,
	#					#mode='FAST_COMPILE',
	#					)
	
	#time3
	print 'calculate'
	time3 = time.clock()
	random_row0 = np.random.randint(0, nword, length)
	#print indexfunction(random_mat0.toarray())
	p,index = pfunction(random_row0)
	print p
	print get_index(index, 0, length-1)
	#for i in range(length): print v[i,i]
	#print vfunction(random_mat0.toarray())
	
	#end
	end_time = time.clock()
	print 'start', time1 - start_time, 'model', time2 - time1, 'functon', time3 - time2, 'calculate', end_time - time3, 'end'
	print end_time - start_time
	
	#theano.printing.pydotprint(function, outfile = 'pf.png', var_with_name_simple=True)
	'''
	print 'gparams...'
	start_time = time.clock()
	#pf_line = pf[0]
	cost = pf[0, end]
	gparams = T.grad(cost, model.params, consider_constant=[model.L, vec0, end]) 
	updates = [
			(param, param - 0.001 * gparam)
			for param, gparam in zip(model.params, gparams)
			]
	end_time = time.clock()
	print 'gparams done', end_time - start_time
	
	print 'cost model...'
	start_time = time.clock()
	cost_model = theano.function(
							inputs=[vec0, end],
			    			outputs=cost,
			        		mode=theano.Mode(linker='cvm', optimizer='fast_compile'),
			        		updates=updates,
			        		on_unused_input='warn'
			        		)
	end_time = time.clock()
	print 'cost model done', end_time - start_time
	#theano.printing.pydotprint(updates_model, outfile = 'updates.png', var_with_name_simple=True)
	'''