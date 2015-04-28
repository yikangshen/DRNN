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

compile_mode = theano.Mode(linker='cvm', optimizer='fast_compile')

max_length = 60

class DRNN(object):
	def __init__(self, rng, size, N_word, max_length, Wf=None, Wp=None, L=None, activation = T.tanh):
		self.size = size
		self.max_length = max_length
		
		#initial Wf
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
		
		#initial Wp
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
		
		v1 = T.vector('v1')
		v2 = T.vector('v2')
		dv = T.vector('dv')
		p1 = T.scalar('p1')
		p2 = T.scalar('p2')
		dp = T.scalar('dp')
		i = T.iscalar('i')
		
		f_function = self.f_function(v1, v2)
		p_function = self.p_function(v1, v2, p1, p2)
		L_i_functin = self.L_i_function(i)
		
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
								outputs=L_i_functin
								)
		#g_f = [
		#	T.dot(dv, theano.gradient.jacobian(f_function, element))
		#	for element in [v1, v2]
		#	]
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
		
		self.p_matrix = np.ones([self.max_length, self.max_length], dtype = np.float32) * (-1.0)
		self.v_matrix = np.zeros([self.max_length, self.max_length, self.size], dtype = np.float32)
		self.ind_matrix = np.zeros([self.max_length, self.max_length], dtype = np.int32)
		self.pf_dict = {}
		self.vf_dict = {}
		
	def f_function(self, v1, v2):
		return T.tanh(T.dot(self.Wf, T.concatenate([v1, v2, [np.float32(1.0)]])))
	
	def p_function(self, v1, v2, p1, p2):
		return T.nnet.sigmoid(T.dot(self.Wp, T.concatenate([v1, v2, [np.float32(1.0)]]))) * p1 * p2
	
	def L_i_function(self, i):
		return self.L[i]
	
	def get_pv(self, sentence):
		self.p_matrix.fill(-1.0)
		self.v_matrix.fill(0.0)
		self.ind_matrix.fill(0)
		return self._get_pv(sentence, 0, sentence.shape[0]-1)
	
	def _get_pv(self, sentence, starti, endi):
		if self.p_matrix[starti, endi] >= 0:
			return (
				self.p_matrix[starti, endi], 
				self.v_matrix[starti, endi],
				)
		
		if starti == endi:
			self.p_matrix[starti, endi] = 1.0
			self.v_matrix[starti, endi] = self.L_i(sentence[starti])
			self.ind_matrix[starti, endi] = 0
			return np.float32(1.0), self.v_matrix[starti, endi].astype(np.float32)
		
		max_p = 0.0
		max_i = 0
		for i in xrange(starti, endi):
			new_p1, new_v1 = self._get_pv(sentence, starti, i)
			new_p2, new_v2 = self._get_pv(sentence, i+1, endi)
			
			new_p = self.p(new_v1, new_v2, new_p1, new_p2)
			
			if new_p > max_p:
				max_p = new_p
				max_i = i
				
		new_p1, new_v1 = self._get_pv(sentence, starti, max_i)
		new_p2, new_v2 = self._get_pv(sentence, max_i+1, endi)
		max_v = self.f(new_v1, new_v2)
		
		self.p_matrix[starti, endi] = max_p
		self.v_matrix[starti, endi] = max_v
		self.ind_matrix[starti, endi] = max_i
		
		return max_p, max_v
	
	def get_pv_function(self, i, j, sentence_tensor):
		if i < j:
			k = self.ind_matrix[i,j]
			p1f, v1f = self.get_pv_function(i, k, sentence_tensor)
			p2f, v2f = self.get_pv_function(k+1, j, sentence_tensor)
			pf = self.p_function(v1f, v2f, p1f, p2f)
			vf = self.f_function(v1f, v2f)
			return pf, vf
		elif i == j:
			return 1.0, self.L_i_function(sentence_tensor[i])
		else:
			raise 'error i > j'
		
	def backprop(self, i, j, dp, dv):
		if i < j:
			k = self.ind_matrix[i,j]
			v = self.v_matrix[i,j]
			
			p1 = self.p_matrix[i,k]
			p2 = self.p_matrix[k+1,j]
			v1 = self.v_matrix[i,k]
			v2 = self.v_matrix[k+1,j]
			
			[dWp, dv1p, dv2p, dp1, dp2] = self.g_p(v1, v2, p1, p2, dp)
			
			dva = 1 - v**2
			da = dva * dv
			dWf = np.outer(da, np.concatenate([v1, v2, [np.float32(1.0)]]))
			#dWf = dvw * dv.reshape(self.size,1)
			[dv1f, dv2f] = self.g_f(da)
			
			dWp1, dWf1 = self.backprop(i, k, dp1, dv1p + dv1f)
			dWp2, dWf2 = self.backprop(k+1, j, dp2, dv2p + dv2f)
			
			return dWp + dWp1 + dWp2, dWf + dWf1 + dWf2
		elif i == j:
			return 0.0, 0.0
		else:
			raise 'error i > j'
				
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
	vec0 = T.ivector('vec0')
	vec1 = T.ivector('vec1')
	
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
				
				p1, v1 = model.get_pv(sentence)
				p1f, v1f = model.get_pv_function(0, sentence.shape[0]-1, vec0)
				index_list = get_index(model.ind_matrix, 0, sentence.shape[0]-1)
				p2, v2 = model.get_pv(random_row)
				p2f, v2f = model.get_pv_function(0, sentence.shape[0]-1, vec1)
				
				if p1 - p2 > margin:
					#cost = 0
					gparams_value = [0, 0]
				else:
					#cost = -(p1 - p2 - margin)
					p_difference = p1f - p2f
					cost = (
						T.max([0, -(p_difference - margin)])
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
						
				print n, 'p:', p1, ',', p2, 'p1-p2:', p1 - p2, 'p1/p2:', p1 / p2
				print index_list
				differencelist.append(p1 - p2)
				
			for i in range(len(gps)):
				model.params[i].set_value(model.params[i].get_value() - learning_rate * gps[i] / batch_size)
			
		print 'epoch:', epoch, 'p0, p1 difference:', np.mean(differencelist)
		average_diff.append(np.mean(differencelist))
		
		print average_diff
		
		params = []
		for param in model.params:
			params.append(param.get_value())
		output = open('params_epoch_'+str(epoch)+'.pkl', 'w')
		cPickle.dump(params, output)
		output.close()

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
	length = 60
	
	rng = np.random.RandomState(1234)
	model = DRNN(rng, L.shape[1], nword, length, L=L)
	
	#start
	print 'start'
	start_time = time.clock()
	
	vec = T.ivector(name = 'vec')
	random_row0 = np.random.randint(0, nword, length)
	p,v = model.get_pv(random_row0)
	pf, vf = model.get_pv_function(0, length-1, vec)
	print p
	print get_index(model.ind_matrix, 0, length-1)
	
	#random_row1 = np.random.randint(0, nword, length)
	#p1,v1 = model.get_pv(random_row0)
	#pf1, vf1 = model.get_pv_function(0, length-1, vec)
	
	#p_model = theano.function(
	#						inputs=[vec],
	#		    			outputs=pf,
	#		        		#mode=compile_mode,
	#		        		on_unused_input='warn'
	#		        		)
	#print p_model(random_row0)
	#print p_model(random_row1)
	
	time1 = time.clock()
	
	print 'train_model...'
	gparams = T.grad(pf, model.params)
	
	gparams_model = theano.function(
								inputs=[vec],
			    				outputs=gparams,
			        			mode='FAST_COMPILE',
			        			on_unused_input='warn'
			        			)
	[dWf0, dWp0] = gparams_model(random_row0)
	#print dWp0
	
	time2 = time.clock()
	
	print 'backprop...'
	dWp1, dWf1 = model.backprop(0, length-1, np.float32(1.0), np.zeros(model.size, dtype=np.float32))
	print dWp1

	end_time = time.clock()
	
	#end
	print 'end'
	print dWf0/dWf1
	print (dWp0/dWp1).max(), (dWp0/dWp1).min()
	print (dWf0/dWf1).max(), (dWf0/dWf1).min()
	print 'start', time1 - start_time, 'train_model', time2 - time1, 'backprop', end_time - time2, 'end'
	print end_time - start_time
	