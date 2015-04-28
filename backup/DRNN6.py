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

max_length = 60

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
		
		v1 = T.fvector('v1')
		v2 = T.fvector('v2')
		dv = T.fvector('dv')
		p1 = T.fscalar('p1')
		p2 = T.fscalar('p2')
		dp = T.fscalar('dp')
		i = T.iscalar('i')
		
		f_function = self.f_function(v1, v2)
		p_function = self.g_function(v1, v2) * p1 * p2
		
		self.f = theano.function(
								inputs=[v1, v2],
								outputs=f_function
								)
		self.p = theano.function(
								inputs=[v1, v2, p1, p2],
								outputs=p_function
								)
		self.L_i = theano.function(
								inputs=[i, ],
								outputs=self.L[i]
								)
		
		g_f = [
			T.dot(dv, self.Wf[:, 0:self.size]),
			T.dot(dv, self.Wf[:, self.size:self.size*2])
			]
		g_p = [
			T.grad(p_function, element) * dp
			for element in [self.Wp, v1, v2, p1, p2]
			]
		
		self.g_p = theano.function(
								inputs=[v1, v2, p1, p2, dp],
								outputs=g_p
								)
		
		self.g_f = theano.function(
								inputs=[dv],
								outputs=g_f
								)
		

	def f_function(self, v1, v2):
		return T.tanh(T.dot(self.Wf, T.concatenate([v1, v2, [np.float32(1.0)]])))
	
	def g_function(self, v1, v2):
		return T.nnet.sigmoid(T.dot(self.Wp, T.concatenate([v1, v2, [np.float32(1.0)]])))
	
	#p_scalar_function, v_vector_function = pv_function(sentence_matrix)
	def pv_function(self, tensor_input):
		indexf_matrix = theano.shared(
									np.zeros(
										[self.max_length, self.max_length], 
										dtype=np.int32
										),
									name = 'indexf_matrix',
									borrow=True
									)
		
		pf_matrix = theano.shared(
								np.zeros(
										[self.max_length, self.max_length], 
										dtype=theano.config.floatX
										),
								name = 'pf_matrix',
								borrow=True
								)
		pf_matrix = T.set_subtensor(pf_matrix[0, 0:tensor_input.shape[0]], 1.0)
		
		vf_matrix = theano.shared(
								np.zeros(
										(self.max_length, self.max_length, self.size), 
										dtype=theano.config.floatX
										),
								name = 'vf_matrix',
								borrow=True
								)
		results, updates = theano.map(
				fn = lambda i, L, t_tensor_input: L[t_tensor_input[i]],
				sequences=[T.arange(tensor_input.shape[0])],
				non_sequences=[self.L, tensor_input],
				name = 'vf_matrix prepare'
				)
		vf_matrix = T.set_subtensor(vf_matrix[0, 0:tensor_input.shape[0]], results)
		
		for i in range(1,self.max_length):
			results, updates = theano.map(
				fn = self._pv_function,
				sequences=[T.arange(self.max_length-i)],
				non_sequences = [i, pf_matrix, vf_matrix],
				#name = 'pv function'
				)
			
			indexf_matrix = T.set_subtensor(indexf_matrix[i, 0:self.max_length-i], results[0])
			pf_matrix = T.set_subtensor(pf_matrix[i, 0:self.max_length-i], results[1])
			vf_matrix = T.set_subtensor(vf_matrix[i, 0:self.max_length-i], results[2])
			
		return indexf_matrix, pf_matrix, vf_matrix
	
	def _pv_function(self, tensor_left, length, pf_matrix, vf_matrix):
		tensor_right = tensor_left + length
		results, updates = theano.map(
				fn = self.get_new_p,
				sequences=[T.arange(tensor_left, tensor_right)],
				non_sequences = [tensor_left, tensor_right, pf_matrix, vf_matrix],
				name = 'pv function'
				)
		max_pf, index = T.max_and_argmax(results)
		max_vf = self.get_new_v(index + tensor_left, tensor_left, tensor_right, vf_matrix)
		return [index + tensor_left, max_pf, max_vf]
	
	def get_new_p(self, i, tensor_left, tensor_right, pf_matrix, vf_matrix):
		p1f = pf_matrix[i - tensor_left, tensor_left]
		v1f = vf_matrix[i - tensor_left, tensor_left]
		p2f = pf_matrix[tensor_right - i - 1, i+1]
		v2f = vf_matrix[tensor_right - i - 1, i+1]
		
		new_pf = self.g_function(v1f, v2f) * p1f * p2f
		
		return new_pf
	
	def get_new_v(self, i, tensor_left, tensor_right, vf_matrix):
		v1f = vf_matrix[i - tensor_left, tensor_left]
		v2f = vf_matrix[tensor_right - i - 1, i+1]
		
		new_vf = self.f_function(v1f,v2f)
		
		return new_vf
	
	def get_pv_function(self, index_matrix, i, j, sentence_tensor):
		if i < j:
			k = index_matrix[j-i, i]
			p1f, v1f = self.get_pv_function(index_matrix, i, k, sentence_tensor)
			p2f, v2f = self.get_pv_function(index_matrix, k+1, j, sentence_tensor)
			pf = self.g_function(v1f, v2f) * p1f * p2f
			vf = self.f_function(v1f, v2f)
			return pf, vf
		elif i == j:
			return 1.0, self.L[sentence_tensor[i]]
		else:
			raise 'error i > j'
		
	def backprop(self, i, j, dp, dv, ind_matrix, p_matrix, v_matrix):
		if i < j:
			k = ind_matrix[j-i,i]
			v = v_matrix[j-i,i]
			
			p1 = p_matrix[k-i,i]
			p2 = p_matrix[j-(k+1),k+1]
			v1 = v_matrix[k-i,i]
			v2 = v_matrix[j-(k+1),k+1]
			
			[dWp, dv1p, dv2p, dp1, dp2] = self.g_p(v1, v2, p1, p2, dp)
			
			dva = 1 - v**2
			da = dva * dv
			dWf = np.outer(da, np.concatenate([v1, v2, [np.float32(1.0)]]))
			#dWf = dvw * dv.reshape(self.size,1)
			[dv1f, dv2f] = self.g_f(da)
			
			dWp1, dWf1 = self.backprop(i, k, dp1, dv1p + dv1f, ind_matrix, p_matrix, v_matrix)
			dWp2, dWf2 = self.backprop(k+1, j, dp2, dv2p + dv2f, ind_matrix, p_matrix, v_matrix)
			
			return dWp + dWp1 + dWp2, dWf + dWf1 + dWf2
		elif i == j:
			return 0.0, 0.0
		else:
			raise 'error i > j'
	
	def get_index(self, index_matrix, i, j):
		if i < j:
			k = index_matrix[j-i, i]
			return [
				k.tolist(), 
				self.get_index(index_matrix, i, k), 
				self.get_index(index_matrix, k+1, j)
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
	vec0 = T.ivector('vec0')
	vec1 = T.ivector('vec1')
	#start = T.iscalar('start')
	end = T.iscalar('end')
	
	print 'model...'
	indexf0, pf0, vf0 = model.pv_function(vec0)
	indexf1, pf1, vf1 = model.pv_function(vec1)
				
	compile_mode = theano.Mode(linker='cvm', optimizer='fast_compile')
	
	print 'function...'
	pi0_function = theano.function(
								inputs=[vec0, end],
			    				outputs=[pf0[end, 0], indexf0],
			        			#mode=compile_mode,
			        			on_unused_input='warn'
			        			)
	
	pi1_function = theano.function(
								inputs=[vec1, end],
			    				outputs=[pf1[end, 0], indexf1],
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
				
				[p0, index_matrix0] = pi0_function(sentence, sentence.shape[0]-1)
				#p0 = p0 * (2**(sentence.shape[0]-1))
				index_list = model.get_index(index_matrix0, 0, sentence.shape[0]-1)
				[p1, index_matrix1] = pi1_function(random_row, sentence.shape[0]-1)
				#p1 = p1 * (2**(sentence.shape[0]-1))
				
				if p0 - p1 > margin:
					#cost = 0
					gparams_value = [0, 0]
				else:
					pf_vec0, vf_vec0 = model.get_pv_function(index_matrix0, 0, sentence.shape[0]-1, vec0)
					pf_vec1, vf_vec1 = model.get_pv_function(index_matrix1, 0, sentence.shape[0]-1, vec1)
					cost = (
						-(pf_vec0 - pf_vec1 - margin)
						+ L1_reg * model.L1
				        + L2_reg * model.L2_sqr
				        )
					gparams = T.grad(cost, model.params, disconnected_inputs='warn') 
					
					gparams_model = theano.function(
												inputs=[vec0, vec1],
							    				outputs=gparams,
							        			mode='FAST_COMPILE',
							        			on_unused_input='warn'
							        			)
					gparams_value = gparams_model(sentence, random_row)
					
				if gps == None:
					gps = gparams_value
				else:
					for i in range(len(gps)):
						gps[i] = gps[i] + gparams_value[i]
						
				print n, 'p:', p0, ',', p1, 'p0-p1:', p0 - p1, 'p0/p2:', p0 / p1
				print index_list
				differencelist.append(p0 - p1)
				
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
	print 'function'
	time2 = time.clock()
	#indexfunction = pfunction = theano.function(
	#						inputs=[mat0],
	#						outputs=indexf,
	#						#mode='FAST_COMPILE',
	#						)
	pfunction = theano.function(
							inputs=[vec0],
							outputs=[indexf, pf, vf],
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
	index, p, v = pfunction(random_row0)
	print p
	print index
	print model.get_index(index, 0, length-1)
	#for i in range(length): print v[i,i]
	#print vfunction(random_mat0.toarray())
	#end
	end_time = time.clock()
	print 'start', time1 - start_time, 'model', time2 - time1, 'function', time3 - time2, 'calculate', end_time - time3, 'end'
	print end_time - start_time
	
	time1 = time.clock()
	
	print 'train_model...'
	pf_vec0, vf_vec0 = model.get_pv_function(index, 0, length-1, vec0)
	gparams = T.grad(pf_vec0, model.params)
	
	gparams_model = theano.function(
								inputs=[vec0],
			    				outputs=gparams,
			        			mode='FAST_COMPILE',
			        			on_unused_input='warn'
			        			)
	[dWf0, dWp0] = gparams_model(random_row0)
	#print dWp0
	
	time2 = time.clock()
	print 'train_model', time2 - time1
	
	time1 = time.clock()
	
	print 'backprop...'
	dWp1, dWf1 = model.backprop(0, length-1, np.float32(1.0), np.zeros(model.size, dtype=np.float32), index, p, v)
	
	time2 = time.clock()
	print 'backprop', time2 - time1
	print (dWp0/dWp1-1).max(), (dWf0/dWf1-1).max()
	print np.abs(dWf0-dWf1).mean()
	print np.abs(dWf0).mean()
	
	start_time = time.clock()
	'''
	print 'gparams...'
	#pf_line = pf[0]
	cost = pf[end, 0]
	gparams = T.grad(cost, model.params) 
	print 'gparams done'
	
	time1 = time.clock()
	
	print 'cost model...'
	cost_model = theano.function(
							inputs=[vec0, end],
			    			outputs=gparams,
			        		#mode=theano.Mode(linker='cvm', optimizer='fast_compile'),
			        		on_unused_input='warn'
			        		)

	print 'cost model done', end_time - start_time
	
	time2 = time.clock()
	
	print 'calculate...'
	cost_model(random_row0, length-1)
	print 'calculate done'
	
	time3 = time.clock()
	
	print 'gparams', time1 - start_time, 'cost model', time2 - time1, 'calculate', time3 - time2, 'end'
	#theano.printing.pydotprint(updates_model, outfile = 'updates.png', var_with_name_simple=True)
	'''