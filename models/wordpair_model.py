import tensorflow as tf
import numpy as np
import pickle
from layers import embeddings_layer
from layers import mlp_layer
from helpers import io_helper

class WordPairModel(object):
	"""
	A model for predicting relations/scores between pairs of words by means of non-linear (MLP) transformation of input word representations
	"""
	def __init__(self, embeddings, embedding_size, mlp_hidden_layer_sizes, same_mlp = True, activation = tf.nn.tanh, distance_measure = "cosine", scope = "word_pair_model"):
		self.embeddings = embeddings
		self.scope = scope
		self.same_mlp = same_mlp
		self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
		self.distance_measure = distance_measure

		with tf.name_scope(self.scope + "__placeholders"):
			self.input_w1 = tf.placeholder(tf.int32, [None,], name="w1")
			self.input_w2 = tf.placeholder(tf.int32, [None,], name="w2")
			self.dropout = tf.placeholder(tf.float64, name="dropout")
		
		self.emb_layer = embeddings_layer.EmbeddingLayer(None, self.embeddings, embedding_size, update_embeddings = False)
		
		# looking up word embeddings from their vocabulary indices
		self.embs_w1 = self.emb_layer.lookup(self.input_w1)
		self.embs_w2 = self.emb_layer.lookup(self.input_w2)

		# MLPs (with or without shared parameters, depending on same_mapper)
		print("Defining first MLP...")
		self.mlp1 = mlp_layer.MultiLayerPerceptron(mlp_hidden_layer_sizes, embedding_size, scope = "mlp_encoder" + ("" if same_mlp else "_first"), unique_scope_addition = "_1")
		self.mlp1.define_model(activation = activation, previous_layer = self.embs_w1, share_params = None) 
		
		print("Defining second MLP...")	
		self.mlp2 = mlp_layer.MultiLayerPerceptron(mlp_hidden_layer_sizes, embedding_size, scope = "mlp_encoder" + ("" if same_mlp else "_second"), unique_scope_addition = "_2")
		self.mlp2.define_model(activation = activation, previous_layer = self.embs_w2, share_params = same_mlp)

		#self.mlp1.outputs = tf.Print(self.mlp1.outputs, [self.mlp1.outputs], message = "First mapped vectors\n")
		#self.mlp2.outputs = tf.Print(self.mlp2.outputs, [self.mlp2.outputs], message = "Second mapped vectors\n")

		if self.distance_measure == "cosine":
			self.norm1 = tf.nn.l2_normalize(self.mlp1.outputs, dim = [1])
			self.norm2 = tf.nn.l2_normalize(self.mlp2.outputs, dim = [1])
			self.outputs = tf.constant(1.0, dtype = tf.float64) - tf.reduce_sum(tf.multiply(self.norm1, self.norm2), axis = 1)

		elif self.distance_measure == "euclidean":
			self.outputs = tf.norm(self.mlp1.outputs - self.mlp2.outputs, axis = 1)	

		else: 
			raise ValueError("Unknown distance function")

		# final outputs are also prediction scores for word pairs
		self.preds = self.outputs
		#self.preds = tf.Print(self.preds, [self.preds], message = "Batch predictions\n")


		# the l2_loss (for regularization) of the model is the sum of l2_norms of all parameters
		self.l2_loss = (self.mlp1.l2_loss if same_mlp else (self.mlp2.l2_loss + self.mlp1.l2_loss))
			

	def define_optimization(self, loss_function, dist_reg_factor, l2_reg_factor = 0.01, learning_rate = 1e-3, loss_function_params = None):
		print("Defining loss...")
		with tf.name_scope(self.scope + "__placeholders"):
			self.input_y = tf.placeholder(tf.float64, [None], name="input_y")
		if loss_function_params:
			self.pure_loss = loss_function(self.outputs, self.input_y, loss_function_params)
		else:
			self.pure_loss = loss_function(self.outputs, self.input_y)

		if self.distance_measure == "cosine":
			cosines1 = tf.constant(1.0, dtype = tf.float64) - tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(self.mlp1.outputs, dim = [1]), tf.nn.l2_normalize(self.embs_w1, dim = [1])), axis = 1)
			cosines2 = tf.constant(1.0, dtype = tf.float64) - tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(self.mlp2.outputs, dim = [1]), tf.nn.l2_normalize(self.embs_w2, dim = [1])), axis = 1)
			self.distance_loss = tf.reduce_sum(cosines1) + tf.reduce_sum(cosines2)
			
		elif self.distance_measure == "euclidean": 
			self.distance_loss = tf.nn.l2_loss(tf.subtract(self.mlp1.outputs, self.embs_w1)) + tf.nn.l2_loss(tf.subtract(self.mlp2.outputs, self.embs_w2))
		
		else: 	
			raise ValueError("Unknown distance function")
		
		self.loss = self.pure_loss + dist_reg_factor * self.distance_loss
		self.loss += l2_reg_factor * self.l2_loss
			
		print("Defining optimizer...")
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
		print("Done!...")
				
	def get_feed_dict(self, left_words, right_words, labels, dropout):
		fd = { self.input_w1 : left_words, self.input_w2 : right_words, self.input_y : labels, self.dropout : dropout}
		fd.update(self.mlp1.get_feed_dict(None, None, dropout))
		fd.update(self.mlp2.get_feed_dict(None, None, dropout))
		return fd

	def get_variable_values(self, session):
		variables = []
		first_mlp_params = self.mlp1.get_variable_values(session)
		variables.append(first_mlp_params)

		if not self.same_mlp:
			second_mlp_params = self.mlp2.get_variable_values(session)
			variables.append(second_mlp_params)
		return variables

	def set_variable_values(self, session, variables):
		val = variables.pop(0)
		self.mlp1.set_variable_values(session, val)
		
		if not self.same_mlp:
			val = variables.pop(0)
			self.mlp2.set_variable_values(session, val)

		if len(variables) > 0:
			raise ValueError("Not all variables have been assigned!")

	def get_hyperparameters(self):
		return [self.same_mlp, self.mlp_hidden_layer_sizes, self.distance_measure]

	def get_model(self, session):
		hyperparams = self.get_hyperparameters()
		variables = self.get_variable_values(session)
		return (hyperparams, variables)

	def save_model(self, session, path):
		io_helper.serialize(self.get_model(session), path)	