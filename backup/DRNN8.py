'''
2015.3.12
by hmwv1114
'''

import numpy as np

import theano
import theano.tensor as T

import cPickle

import time
import sys, os

from multiprocessing import Pool

#sys.setrecursionlimit(10000)

max_length = 60

class DRNN(object):
	def __init__(self, rng, size, N_word, max_length, Wf_values=None, Wp_values=None, L_values=None, activation = T.tanh):
		self.size = size
		self.max_length = max_length
		
		#initial Wf, bf
		if Wf_values is None:
			Wf_values = np.asarray(
								rng.uniform(
										low=-np.sqrt(6. / (size + size*2)),
										high=np.sqrt(6. / (size + size*2)),
										size=(size, size*2+1)
										),
								dtype=theano.config.floatX
								)
			if activation == T.nnet.sigmoid:
				Wf_values *= 4

		Wf = theano.shared(value=Wf_values, name='Wf', borrow=True)

		self.Wf = Wf
		
		#initial Wp, bp
		if Wp_values is None:
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
		if L_values is None:
			L_values = np.asarray(
					rng.uniform(
					low=-np.sqrt(6. / (N_word)),
                    high=np.sqrt(6. / (N_word)),
                    size=(N_word, size)
                ),
                dtype=theano.config.floatX
            )
			
			self.L = theano.shared(value=L_values, name='L', borrow=True)
			
			self.params = [
						self.Wf,
						self.Wp,
						self.L
						]
		else:
			self.L = theano.shared(value=L_values, name='L', borrow=True)
		
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
		v = T.fvector('v')
		p = T.fscalar('p')
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
		da = (1 - v**2) * dv
		dWf = T.outer(da, T.concatenate([v1, v2, [np.float32(1.0)]]))
		g_f = [
			dWf,
			T.dot(da, self.Wf[:, 0:self.size]),
			T.dot(da, self.Wf[:, self.size:self.size*2])
			]
		#g_p = [
		#	T.grad(p_function, element) * dp
		#	for element in [self.Wp, v1, v2, p1, p2]
		#	]
		b = p / p1 / p2
		db = b * (1 - b)
		temp = dp * p1 * p2 * db
		g_p = [
			temp * T.concatenate([v1, v2, [np.float32(1.0)]]),
			temp * self.Wp[0:self.size],
			temp * self.Wp[self.size:self.size*2],
			dp * p / p1,
			dp * p / p2
			]
		
		self.g_p = theano.function(
								inputs=[v1, v2, p1, p2, p, dp],
								outputs=g_p
								)
		
		self.g_f = theano.function(
								inputs=[v1, v2, v, dv],
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
		
		[indexf_matrix, pf_matrix, vf_matrix], updates = theano.reduce(
				fn = self._pv_row,
				sequences=[T.arange(1,self.max_length)],
				outputs_info=[indexf_matrix, pf_matrix, vf_matrix],
				#name = 'pv function'
				)
			
		return indexf_matrix, pf_matrix, vf_matrix
	
	def _pv_row(self, i, indexf_matrix, pf_matrix, vf_matrix):
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
				fn = self.get_new_pf,
				sequences=[T.arange(start=tensor_left, stop=tensor_right)],
				non_sequences = [tensor_left, tensor_right, pf_matrix, vf_matrix],
				name = 'pv function'
				)
		max_pf, index = T.max_and_argmax(results, axis=0)
		max_vf = self.get_new_vf(index + tensor_left, tensor_left, tensor_right, vf_matrix)
		return [index + tensor_left, max_pf, max_vf]
	
	def get_new_pf(self, i, tensor_left, tensor_right, pf_matrix, vf_matrix):
		p1f = pf_matrix[i - tensor_left, tensor_left]
		v1f = vf_matrix[i - tensor_left, tensor_left]
		p2f = pf_matrix[tensor_right - i - 1, i+1]
		v2f = vf_matrix[tensor_right - i - 1, i+1]
		
		new_pf = self.g_function(v1f, v2f) * p1f * p2f
		
		return new_pf
	
	def get_new_vf(self, i, tensor_left, tensor_right, vf_matrix):
		v1f = vf_matrix[i - tensor_left, tensor_left]
		v2f = vf_matrix[tensor_right - i - 1, i+1]
		
		new_vf = self.f_function(v1f,v2f)
		
		return new_vf
	
	def pv_value(self, sentence):
		p_matrix = np.ones([self.max_length, self.max_length], dtype = np.float32) * (-1.0)
		v_matrix = np.zeros([self.max_length, self.max_length, self.size], dtype = np.float32)
		ind_matrix = np.zeros([self.max_length, self.max_length], dtype = np.int32)
		self._pv_value(sentence, 0, sentence.shape[0]-1, ind_matrix, p_matrix, v_matrix)
		return ind_matrix, p_matrix, v_matrix
	
	def _pv_value(self, sentence, starti, endi, ind_matrix, p_matrix, v_matrix):
		length = endi - starti
		if p_matrix[length, starti] >= 0:
			return (
				p_matrix[length, starti], 
				v_matrix[length, starti],
				)
		
		if starti == endi:
			p_matrix[length, starti] = 1.0
			v_matrix[length, starti] = self.L_i(sentence[starti])
			ind_matrix[length, starti] = 0
			return np.float32(1.0), v_matrix[length, starti].astype(np.float32)
		
		max_p = 0.0
		max_i = 0
		for i in xrange(starti, endi):
			new_p1, new_v1 = self._pv_value(sentence, starti, i, ind_matrix, p_matrix, v_matrix)
			new_p2, new_v2 = self._pv_value(sentence, i+1, endi, ind_matrix, p_matrix, v_matrix)
			
			new_p = self.p(new_v1, new_v2, new_p1, new_p2)
			
			if new_p > max_p:
				max_p = new_p
				max_i = i
				
		new_p1, new_v1 = self._pv_value(sentence, starti, max_i, ind_matrix, p_matrix, v_matrix)
		new_p2, new_v2 = self._pv_value(sentence, max_i+1, endi, ind_matrix, p_matrix, v_matrix)
		max_v = self.f(new_v1, new_v2)
		
		p_matrix[length, starti] = max_p
		v_matrix[length, starti] = max_v
		ind_matrix[length, starti] = max_i
		
		return max_p, max_v
	
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
			p = p_matrix[j-i,i]
			
			p1 = p_matrix[k-i,i]
			p2 = p_matrix[j-(k+1),k+1]
			v1 = v_matrix[k-i,i]
			v2 = v_matrix[j-(k+1),k+1]
			
			[dWp, dv1p, dv2p, dp1, dp2] = self.g_p(v1, v2, p1, p2, p, dp)
			[dWf, dv1f, dv2f] = self.g_f(v1, v2, v, dv)
			
			dWp1, dWf1 = self.backprop(i, k, dp1, dv1p + dv1f, ind_matrix, p_matrix, v_matrix)
			dWp2, dWf2 = self.backprop(k+1, j, dp2, dv2p + dv2f, ind_matrix, p_matrix, v_matrix)
			
			return dWp + dWp1 + dWp2, dWf + dWf1 + dWf2
		elif i == j:
			return 0.0, 0.0
		else:
			raise 'error i > j'
	
	def get_parse_tree(self, index_matrix, i, j):
		if i < j:
			k = index_matrix[j-i, i]
			return [
				k.tolist(), 
				self.get_parse_tree(index_matrix, i, k), 
				self.get_parse_tree(index_matrix, k+1, j)
				]
		elif i == j:
			return [i]
		else:
			raise 'error i > j'
				
if __name__ == '__main__':
	dictfile = open('word_dict_stanford.pkl', 'r')
	word_dict, L = cPickle.load(dictfile)
	dictfile.close()
	nword = len(word_dict.keys())

	length = 60
	rng = np.random.RandomState(1234)
	model = DRNN(rng, L.shape[1], nword, max_length, L_values=L)
	random_row0 = np.random.randint(0, nword, max_length)
	random_row1 = np.random.randint(0, nword, max_length)
	
	
	print 'start pv_value'
	start_time = time.clock()
	vec = T.ivector(name = 'vec')
	index, p, v = model.pv_value(random_row0)
	end_time = time.clock()
	#print index
	print p[length-1, 0]
	print model.get_parse_tree(index, 0, length-1)
	print end_time - start_time
	
	
	#start
	print 'start pv_function'
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
	pfunction = theano.function(
							inputs=[vec0],
							outputs=[indexf, pf, vf],
							)
	
	#time3
	print 'calculate'
	time3 = time.clock()
	#random_row0 = np.random.randint(0, nword, length)
	index, p, v = pfunction(random_row0)
	end_time = time.clock()
	#print p
	print p[length-1, 0]
	#print index
	print model.get_parse_tree(index, 0, length-1)
	#end
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
	