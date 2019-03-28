# -*- coding: utf-8 -*-
import numpy as np
import random

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, sort_len=False):
	""" Generate minibatches of a specific size 

	Arguments:
		inputs -- numpy array of the encoded protein data. Shape: (n_samples, seq_len, n_features)
		targets -- numpy array of the targets. Shape: (n_samples,)
		masks -- numpy array of the protein masks. Shape: (n_samples, seq_len)
		batchsize -- integer, number of samples in each minibatch.
		shuffle -- boolean, shuffle the samples in the minibatches. (default=False)
		sort_len -- boolean, sort the minibatches by sequence length (faster computation, just for training). (default=True) 

	Outputs:
	list of minibatches for protein sequences, targets and masks.
	
	""" 
	assert len(inputs) == len(targets)

	# Calculate the sequence length of each sample
	#len_seq = len(inputs)
	
	# Sort the sequences by length
	if sort_len:
		indices = np.argsort(len_seq)
	else:
		indices = np.arange(len(inputs))

	# Generate minibatches list
	f_idx = len(inputs) % batchsize
	idx_list = list(range(0, len(inputs) - batchsize + 1, batchsize))
	last_idx = None
	if f_idx != 0:
		last_idx = idx_list[-1] + batchsize
		idx_list.append(last_idx)
	
	# Shuffle the minibatches
	if shuffle:
		random.shuffle(idx_list)

	# Split the data in minibatches
	for start_idx in idx_list:
		if start_idx == last_idx:
			rand_samp = batchsize - f_idx
			B = np.random.randint(len(inputs),size=rand_samp)
			excerpt = np.concatenate((indices[start_idx:start_idx + batchsize], B))
		else:
			excerpt = indices[start_idx:start_idx + batchsize]
		#max_prot = np.amax(len_seq[excerpt])

		# Crop batch to maximum sequence length
		if sort_len:
			in_seq = inputs[excerpt][:,:max_prot]
		else:
			in_seq = inputs[excerpt]

		in_target = targets[excerpt]
		shuf_ind = np.arange(batchsize)
				
		# Shuffle samples within each minitbatch
		if shuffle:
			np.random.shuffle(shuf_ind)

		# Return a minibatch of each array
		yield in_seq[shuf_ind], in_target[shuf_ind]