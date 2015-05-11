'''
2015.3.12
by hmwv1114
'''

import numpy as np
from scipy.signal import convolve2d

import cPickle

import time
import sys, os

import gensim.models.word2vec as word2vec

os.environ['MKL_NUM_THREADS'] = '1'

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def softmax(x):
	ex = np.exp(x)
	sumex = ex.sum()
	return ex/sumex

class CNN(object):
	def __init__(self, rng, size, N_word, conv_l=5, n_feature_maps=50, maxk=20, hsize=100, output_size=2, Wc_values=None, Wh_values=None, Ws_values=None, L_values=None, activation = np.tanh):
		self.size = size
		self.N_word = N_word
		self.conv_l = conv_l
		self.n_fmaps = n_feature_maps
		self.maxk = maxk
		self.hsize = hsize
		self.osize = output_size
		
		if Wc_values is None:
			Wc_values = np.asarray(
								rng.uniform(
										low=-np.sqrt(6. / (self.conv_l + self.size)),
										high=np.sqrt(6. / (self.conv_l + self.size)),
										size=(self.n_fmaps, self.conv_l, self.size),
										),
								dtype=np.float32
								)
			if activation == sigmoid:
				Wc_values *= 4

		self.Wc = Wc_values
		
		if Wh_values is None:
			Wh_values = np.asarray(
								rng.uniform(
										low=-np.sqrt(6. / (self.hsize + self.maxk * self.n_fmaps)),
										high=np.sqrt(6. / (self.hsize + self.maxk * self.n_fmaps)),
										size=(self.hsize, self.maxk * self.n_fmaps + 1),
										),
								dtype=np.float32
								)
		self.Wh = Wh_values
		
		if Ws_values is None:
			Ws_values = np.asarray(
								rng.uniform(
										low=-np.sqrt(6. / (self.osize + self.hsize)),
										high=np.sqrt(6. / (self.osize + self.hsize)),
										size=(self.osize, self.hsize + 1),
										),
								dtype=np.float32
								)
		self.Ws = Ws_values
		
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
					self.Wc,
					self.Wh,
					self.Ws
					]
		
		self.L1 = (
            abs(self.Wh).sum()
            + abs(self.Wc).sum()
        )
		
		self.L2_sqr = (
            (self.Wh ** 2).sum()
            + (self.Wc ** 2).sum()
        )
		
		
	def conv(self, mat):
		conv_out = np.zeros([self.n_fmaps, mat.shape[0]-self.conv_l+1])
		for i in range(self.n_fmaps):
			conv_out[i] = convolve2d(mat, self.Wc[i], mode='valid').flatten()
		return np.tanh(conv_out)
	
	def k_max_pooling(self, conv_out):
		pooling_out = np.zeros([self.n_fmaps, self.maxk])
		pooling_out_index = np.zeros_like(conv_out, dtype=np.int32)
		t = np.argsort(conv_out, axis=1)
		for i in range(self.n_fmaps):
			k = 0
			for j in range(conv_out.shape[1]):
				if t[i,j] >= conv_out.shape[1] - self.maxk:
					pooling_out[i,k] = conv_out[i,j]
					pooling_out_index[i,j] = 1
					k += 1
					
		return pooling_out.flatten(), pooling_out_index
	
	def hidden(self, pooling_out):
		return np.tanh(np.dot(self.Wh, np.concatenate([pooling_out, [1.0]])))
	
	def SoftmaxRegression(self, h_out):
		return softmax(np.dot(self.Ws, np.concatenate([h_out, [1.0]])))
	
	def caculate(self, row):
		mat = np.zeros((len(row), self.L.shape[1]))
		for i in range(len(row)):
			mat[i] = self.L[row[i]]
		conv_out = self.conv(mat)
		pooling_out, pooling_out_index = self.k_max_pooling(conv_out)
		h_out = self.hidden(pooling_out)
		l_out = self.SoftmaxRegression(h_out)
		
		return l_out
	
	def bp_SoftmaxRegression(self, dout, h):
		dw = np.outer(dout, np.concatenate([h, [1.0]]))
		dh = np.dot(dout, self.Ws[:, 0:self.hsize])
		
		return dh, dw
		
	def bp_hidden(self, dout, p, h):
		dout = dout * (1-h**2)
		dw = np.outer(dout, np.concatenate([p, [1.0]]))
		dkmax = np.dot(dout, self.Wh[:, 0:self.maxk * self.n_fmaps])
		
		return dkmax, dw
		
	def bp_kmax(self, dout, pooling_out_index):
		dout = dout.reshape([self.n_fmaps, self.maxk])
		dconv_out = np.zeros_like(pooling_out_index, dtype=np.float32)
		for i in range(dconv_out.shape[0]):
			k = 0
			for j in range(dconv_out.shape[1]):
				if pooling_out_index[i,j] > 0:
					dconv_out[i,j] += dout[i,k]
					k += 1
					
		return dconv_out
		
	def bp_conv(self, dout, mat, conv_out):
		dWc = np.zeros_like(self.Wc)
		dout = dout * (1-conv_out**2)
		for i in range(dout.shape[0]):
			for j in range(dout.shape[1]):
				if dout[i,j] != 0:
					dWc[i] += mat[j:j+self.conv_l,:] * dout[i,j]
		
		return dWc
	
	def backprop(self, row, out):
		mat = np.zeros((len(row), self.L.shape[1]))
		for i in range(len(row)):
			mat[i] = self.L[row[i]]
		conv_out = self.conv(mat)
		pooling_out, pooling_out_index = self.k_max_pooling(conv_out)
		h_out = self.hidden(pooling_out)
		l_out = self.SoftmaxRegression(h_out)
		
		arrout = np.zeros(self.osize)
		arrout[out] = 1
		dl = arrout - l_out
		dh, dWs = self.bp_SoftmaxRegression(dl, h_out)
		dkmax, dWh = self.bp_hidden(dh, pooling_out, h_out)
		dconv_out = self.bp_kmax(dkmax, pooling_out_index)
		dWc = self.bp_conv(dconv_out, mat, conv_out)
		
		return dWc, dWh, dWs
				
if __name__ == '__main__':
	model = word2vec.Word2Vec.load('stanford_word_vector')
	nword = len(model.vocab.keys())
	L = model.syn0
	
	length = 60
	rng = np.random.RandomState(1234)
	model = CNN(rng, L.shape[1], nword, L_values=L)
	random_row0 = np.random.randint(0, nword, length)
	

	print 'start caculate'
	start_time = time.clock()
	v = model.caculate(random_row0)
	end_time = time.clock()
	#print index
	print v
	print end_time - start_time
	
	start_time = time.clock()
	for i in range(1):
		dWc, dWh, dWs = model.backprop(random_row0, 1)
		model.Ws += dWs * 0.001
		model.Wh += dWh * 0.001
		model.Wc += dWc * 0.001
		print model.caculate(random_row0)
	end_time = time.clock()
	print end_time - start_time
	
	'''
	time1 = time.clock()
	print 'backprop...'
	dL = np.zeros([nword, L.shape[1]], dtype=np.float32)
	dWp1, dWf1, dL1 = model.backprop(0, length-1, np.float32(1.0), np.zeros(model.size, dtype=np.float32), index, p, v)
	time2 = time.clock()
	print 'backprop', time2 - time1
	n = 0
	for i in range(nword):
		if np.abs(dL1[i]).sum() > 0:
			n += 1
	print n
	'''