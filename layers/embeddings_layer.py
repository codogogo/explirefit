import tensorflow as tf

class EmbeddingLayer(object):
	"""
	A layer containing the Variable for loaded word embeddings. 
	"""
	def __init__(self, vocab_size, pretrained_embs, embedding_size, update_embeddings = False, scope = "embeddings"):
		self.embedding_size = embedding_size
		self.scope = scope
		self.update = update_embeddings

		with tf.variable_scope(self.scope):
			if pretrained_embs is None:
				self.embeddings = tf.get_variable("word_embeddings", shape=[vocab_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
			elif self.update:
				self.embeddings = tf.get_variable("word_embeddings", initializer=pretrained_embs, dtype = tf.float64, trainable = True)
			else:
				self.embeddings = tf.get_variable("word_embeddings", initializer=pretrained_embs, dtype = tf.float64, trainable = False)

	def lookup(self, index_tensor):
		return tf.nn.embedding_lookup(self.embeddings, index_tensor)