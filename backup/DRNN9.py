'''
2015.3.12
by hmwv1114
'''

import numpy as np

import cPickle

import time
import sys, os

from multiprocessing import Pool

os.environ['MKL_NUM_THREADS'] = '1'

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class DRNN(object):
	def __init__(self, rng, size, N_word, Wf_values=None, Wp_values=None, L_values=None, activation = np.tanh):
		self.size = size
		self.N_word = N_word
		
		#initial Wf, bf
		if Wf_values is None:
			Wf_values = np.asarray(
								rng.uniform(
										low=-np.sqrt(6. / (size + size*2)),
										high=np.sqrt(6. / (size + size*2)),
										size=(size, size*2+1)
										),
								dtype=np.float32
								)
			if activation == sigmoid:
				Wf_values *= 4

		self.Wf = Wf_values
		
		#initial Wp, bp
		if Wp_values is None:
			Wp_values = np.asarray(
								rng.uniform(
										low=-np.sqrt(6. / (size*2)),
										high=np.sqrt(6. / (size*2)),
										size=(2*size+1,)
										),
								dtype=np.float32
								)
		self.Wp = Wp_values
		
		if L_values is None:
			L_values = np.asarray(
					rng.uniform(
					low=-np.sqrt(6. / (N_word)),
                    high=np.sqrt(6. / (N_word)),
                    size=(N_word, size)
                ),
                dtype=np.float32
            )
			
		self.L = L_values
			
		self.params = [
					self.Wf,
					self.Wp,
					self.L #L should be the last one
					]
		
		self.L1 = (
            abs(self.Wf).sum()
            + abs(self.Wp).sum()
        )
		
		self.L2_sqr = (
            (self.Wf ** 2).sum()
            + (self.Wp ** 2).sum()
        )
		

	def f(self, v1, v2):
		return np.tanh(np.dot(self.Wf, np.concatenate([v1, v2, [np.float32(1.0)]])))
	
	def g(self, v1, v2):
		return sigmoid(np.dot(self.Wp, np.concatenate([v1, v2, [np.float32(1.0)]])))
	
	def pv_value(self, sentence):
		l = sentence.shape[0]
		p_matrix = np.ones([l, l], dtype = np.float32) * (-1.0)
		v_matrix = np.zeros([l, l, self.size], dtype = np.float32)
		ind_matrix = np.zeros([l, l], dtype = np.int32)
		self._pv_value(sentence, 0, sentence.shape[0]-1, ind_matrix, p_matrix, v_matrix)
		return ind_matrix, p_matrix, v_matrix
	
	def _pv_value(self, sentence, starti, endi, ind_matrix, p_matrix, v_matrix):
		length = endi - starti
		if p_matrix[length, starti] >= 0:
			return p_matrix[length, starti], v_matrix[length, starti]
			
		if starti == endi:
			p_matrix[length, starti] = 1.0
			v_matrix[length, starti] = self.L[sentence[starti]]
			ind_matrix[length, starti] = sentence[starti]
			return np.float32(1.0), v_matrix[length, starti].astype(np.float32)
		
		max_p = 0.0
		max_i = 0
		for i in xrange(starti, endi):
			new_p1, new_v1 = self._pv_value(sentence, starti, i, ind_matrix, p_matrix, v_matrix)
			new_p2, new_v2 = self._pv_value(sentence, i+1, endi, ind_matrix, p_matrix, v_matrix)
			
			new_p = self.g(new_v1, new_v2) * new_p1 * new_p2
			
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
	
	def pv_value2(self, sentence):
		l = sentence.shape[0]
		p_matrix = np.zeros([l, l], dtype = np.float32)
		v_matrix = np.zeros([l, l, self.size], dtype = np.float32)
		ind_matrix = np.zeros([l, l], dtype = np.int32)
		
		for i in range(l):
			p_matrix[0, i] = 1.0
			v_matrix[0, i] = self.L[sentence[i]]
			ind_matrix[0, i] = sentence[i]
		
		for i in range(1, l):
			for j in range(l-i):
				max_p = 0.0
				max_k = 0
				for k in xrange(i):
					new_p1 = p_matrix[k, j]
					new_v1 = v_matrix[k, j]
					new_p2 = p_matrix[i-k-1, j+k+1]
					new_v2 = v_matrix[i-k-1, j+k+1]
					
					new_p = self.g(new_v1, new_v2) * new_p1 * new_p2
					
					if new_p > max_p:
						max_p = new_p
						max_k = k
				new_p1 = p_matrix[max_k, j]
				new_v1 = v_matrix[max_k, j]
				new_p2 = p_matrix[i-max_k-1, j+max_k+1]
				new_v2 = v_matrix[i-max_k-1, j+max_k+1]
				
				max_v = self.f(new_v1, new_v2)
				
				p_matrix[i, j] = max_p
				v_matrix[i, j] = max_v
				ind_matrix[i, j] = max_k + j
				
		return ind_matrix, p_matrix, v_matrix
		
	def backprop(self, i, j, dp, dv, ind_matrix, p_matrix, v_matrix):
		if i < j:
			k = ind_matrix[j-i,i]
			v = v_matrix[j-i,i]
			p = p_matrix[j-i,i]
			
			p1 = p_matrix[k-i,i]
			p2 = p_matrix[j-(k+1),k+1]
			v1 = v_matrix[k-i,i]
			v2 = v_matrix[j-(k+1),k+1]
			
			vv = np.concatenate([v1, v2, [np.float32(1.0)]])
			
			da = (1 - v**2) * dv
			dWf = np.outer(da, vv)
			dv1f = np.dot(da, self.Wf[:, 0:self.size])
			dv2f = np.dot(da, self.Wf[:, self.size:self.size*2])
			
			b = p / p1 / p2
			db = b * (1 - b)
			temp = dp * p1 * p2 * db
			dWp = vv * temp
			dv1p = self.Wp[0:self.size] * temp
			dv2p = self.Wp[self.size:self.size*2] * temp
			dp1 = dp * p / p1
			dp2 = dp * p / p2
			
			dWp1, dWf1, dL1 = self.backprop(i, k, dp1, dv1p + dv1f, ind_matrix, p_matrix, v_matrix)
			dWp2, dWf2, dL2 = self.backprop(k+1, j, dp2, dv2p + dv2f, ind_matrix, p_matrix, v_matrix)
			
			return dWp + dWp1 + dWp2, dWf + dWf1 + dWf2, dL1 + dL2
		elif i == j:
			dL = np.zeros([self.N_word, self.size], dtype=np.float32)
			dL[ind_matrix[i,j]] = dv
			return 0.0, 0.0, dL
		else:
			raise 'error i > j'
	
	def get_parse_tree(self, index_matrix, i, j, sentence = None):
		if i < j:
			k = index_matrix[j-i, i]
			return [
				self.get_parse_tree(index_matrix, i, k, sentence), 
				self.get_parse_tree(index_matrix, k+1, j, sentence)
				]
		elif i == j:
			if sentence != None:
				return sentence[i]
			else:
				return i
		else:
			raise 'error i > j'
		
def pv(args):
	n, model, sentence, random_row = args
	index0, p0, v0 = model.pv_value(sentence)
	index1, p1, v1 = model.pv_value(random_row)
	pp0 = p0[sentence.shape[0]-1,0]
	pp1 = p1[sentence.shape[0]-1,0]
	if pp0 - pp1 > 0.1:
		#cost = 0
		gparams_value = [0, 0]
	else:
		dWp0, dWf0, dL0 = model.backprop(0, sentence.shape[0]-1, -1.0, np.zeros(model.size, dtype=np.float32), index0, p0, v0)
		dWp1, dWf1, dL1 = model.backprop(0, sentence.shape[0]-1, 1.0, np.zeros(model.size, dtype=np.float32), index1, p1, v1)
		gparams_value = [dWf0 + dWf1, dWp0 + dWp1, dL0 + dL1]

	#index_list = model.get_parse_tree(index0, 0, sentence.shape[0]-1)
	#print n, 'p:', pp0, ',', pp1, 'p0-p1:', pp0 - pp1, 'p0/p2:', pp0 / pp1
	#print index_list
	
	return (pp0 - pp1, gparams_value)
				
if __name__ == '__main__':
	dictfile = open('word_dict_stanford.pkl', 'r')
	word_dict, L = cPickle.load(dictfile)
	dictfile.close()
	nword = len(word_dict.keys())

	length = 60
	rng = np.random.RandomState(1234)
	model = DRNN(rng, L.shape[1], nword, L_values=L)
	random_row0 = np.random.randint(0, nword, length)
	random_row1 = np.random.randint(0, nword, length)
	
	print 'start pv_value'
	start_time = time.clock()
	index, p, v = model.pv_value(random_row0)
	end_time = time.clock()
	#print index
	print 'probability:', p[length-1, 0]
	print model.get_parse_tree(index, 0, length-1)
	print end_time - start_time
	
	print 'start pv_value'
	start_time = time.clock()
	index, p, v = model.pv_value2(random_row0)
	end_time = time.clock()
	#print index
	print 'probability:', p[length-1, 0]
	print model.get_parse_tree(index, 0, length-1)
	print end_time - start_time
	
	pool = Pool() 
	start_time = time.clock()
	rs = []
	for i in range(8):
		rs.append((i, model, random_row0, random_row1))
	gparams_values = pool.map(pv, rs) 
	gps = []
	for dpp, gparams_value in gparams_values:
		if gps == None:
			gps = gparams_value
		else:
			for i in range(len(gps)):
				gps[i] + gparams_value[i]
				
		#differencelist.append(dpp)
	end_time = time.clock()
	print 'pool:', end_time - start_time
	
	'''
	time1 = time.clock()
	print 'backprop...'
	dWp1, dWf1 = model.backprop(0, length-1, np.float32(1.0), np.zeros(model.size, dtype=np.float32), index, p, v)
	time2 = time.clock()
	print 'backprop', time2 - time1
	#print (dWp0/dWp1-1).max(), (dWf0/dWf1-1).max()
	#print np.abs(dWf0-dWf1).mean()
	#print np.abs(dWf0).mean()
	'''