import tensorflow as tf
from sys import stdin

def slice_matrix(matrix, indices, col_or_row = "col"):
	slices = []
	indices.append(-1)	
	for i in range(len(indices)):
		start = 0 if i == 0 else indices[i-1]
		size = indices[i] if i == len(indices) - 1 else (indices[i] - start)

		slice = tf.slice(matrix, [0 if col_or_row == "col" else start, start if col_or_row == "col" else 0], [-1 if col_or_row == "col" else size, size if col_or_row == "col" else -1])
		slices.append(slice)
	return slices

def tensor_broadcast(tensor, additional_dims):
	reshape_dims = [1] * len(additional_dims) + tensor.get_shape().as_list()
	tile_dims = additional_dims + ([1] * len(tensor.get_shape().as_list()))
	t_re = tf.reshape(tensor, reshape_dims)
	return tf.tile(t_re, tile_dims)

def broadcast_matmul(larger_dim_tensor, tensor_to_broadcast, none_dim_replacement = None):
	add_dims = (larger_dim_tensor.get_shape().as_list())[: len(larger_dim_tensor.get_shape()) - len(tensor_to_broadcast.get_shape())]
	if (len(add_dims) == 0):
		return tf.matmul(larger_dim_tensor, tensor_to_broadcast)
	else: 
		if add_dims[0] is None and none_dim_replacement is not None:
			add_dims[0] = none_dim_replacement
		smaller_broadcasted = tensor_broadcast(tensor_to_broadcast, add_dims)
		return tf.matmul(larger_dim_tensor, smaller_broadcasted)

def softmax_ignore_zeros(tensor, none_dim_replacement = None):
	nonzero_softmaxes = tf.where(tf.not_equal(tensor, 0), tf.exp(tensor), tensor)

	reshape_dims = tensor.get_shape().as_list()
	if reshape_dims[0] is None and none_dim_replacement is not None:
		reshape_dims[0] = none_dim_replacement
	reshape_dims[-1] = 1

	norms = tf.reshape(tf.reduce_sum(nonzero_softmaxes, axis = 1), reshape_dims)
	return tf.div(nonzero_softmaxes, norms)