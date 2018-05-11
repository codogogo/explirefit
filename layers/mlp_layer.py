import tensorflow as tf

class MultiLayerPerceptron(object):
	"""
	A layer for mapping embeddings from different embedding spaces to the same shared embedding space. 
	"""

	def __init__(self, hidden_layer_sizes, input_size, scope = "mlp_layer", unique_scope_addition = "_1"):
		self.input_size = input_size
		self.hidden_layer_sizes = hidden_layer_sizes
		self.scope = scope
		self.unique_scope_addition = unique_scope_addition
		self.output_size = hidden_layer_sizes[-1]

	def define_model(self, activation = tf.nn.tanh, previous_layer = None, share_params = None):
		self.previous_layer = previous_layer
		with tf.name_scope(self.scope + self.unique_scope_addition + "__placeholders"):
			if previous_layer is None: 
				self.input = tf.placeholder(tf.float64, [None, self.input_size], name = "input_x")
			self.dropout =  tf.placeholder(tf.float64, name="dropout")
		if previous_layer is not None:
			self.input = previous_layer

		self.Ws = []
		self.biases = []
		with tf.variable_scope(self.scope + "__variables", reuse = share_params):
			for i in range(len(self.hidden_layer_sizes)): 
				self.Ws.append(tf.get_variable("W_" + str(i), shape=[(self.input_size if i == 0 else self.hidden_layer_sizes[i-1]), self.hidden_layer_sizes[i]], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64))
				self.biases.append(tf.get_variable("b_" + str(i), initializer=tf.constant(0.1, shape=[self.hidden_layer_sizes[i]], dtype = tf.float64), dtype = tf.float64))
				#self.Ws.append(tf.get_variable("W_" + str(i), initializer=tf.eye(self.input_size if i == 0 else self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i], dtype = tf.float64), dtype = tf.float64))
				#self.biases.append(tf.get_variable("b_" + str(i), initializer=tf.constant(0, shape=[self.hidden_layer_sizes[i]], dtype = tf.float64), dtype = tf.float64))
								

		self.layer_outputs = []
		data_runner = self.input
		for i in range(len(self.Ws)):
			data_runner = tf.nn.dropout(activation(tf.nn.xw_plus_b(data_runner, self.Ws[i], self.biases[i])), self.dropout)
			#data_runner = tf.nn.dropout(tf.nn.xw_plus_b(data_runner, self.Ws[i], self.biases[i]), self.dropout)
			self.layer_outputs.append(data_runner)
		self.outputs = self.layer_outputs[-1]
		
		self.l2_loss = 0
		for i in range(len(self.Ws)):
			self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.Ws[i]) + tf.nn.l2_loss(self.biases[i])

	def define_loss(self, loss_function, l2_reg_factor = 0):
		self.input_y = tf.placeholder(tf.float64, [None, self.hidden_layer_sizes[-1]], name = self.scope + "__input_y")		
		self.preds = tf.nn.dropout(self.outputs, self.dropout)
		self.pure_loss = loss_function(self.preds, self.input_y)
		self.loss = self.pure_loss + l2_reg_factor * self.l2_loss

	def define_optimization(self, learning_rate):
		self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

	def get_feed_dict(self, input_data, labels = None, dropout = 1.0):
		fd_mine = {self.dropout : dropout}
		if self.previous_layer is None:
			fd_mine = { self.input : input_data }
		if labels is not None:
			fd_mine.update({ self.input_y : labels})
		return fd_mine

	def get_variable_values(self, session):
		matrices = []
		biases = []
		for i in range(len(self.Ws)):
			matrices.append(self.Ws[i].eval(session = session))
			biases.append(self.biases[i].eval(session = session))
		return [matrices, biases]

	def set_variable_values(self, session, values):
		if len(values) != 2:
			raise ValueError("Two lists expected, one with values of layer matrices and another with biases")
		for i in range(len(self.Ws)):
			session.run(self.Ws[i].assign(values[0][i]))
			session.run(self.biases[i].assign(values[1][i]))	

	def get_model(self, session):
		return self.get_variable_values(session)