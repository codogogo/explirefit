import numpy as np
import faiss

class Faiss(object):
	def __init__(self, dimension):
		self.dimension = dimension

	def l2_normalize_matrix(self, matrix, matrix_normalized = True):
		if matrix_normalized:
			return matrix
		else:	
			return (matrix / np.linalg.norm(matrix, axis = (1), keepdims = True))


	def index(self, vocabulary, matrix, matrix_normalized = True, measure = "cosine"):
		self.index_vocabulary = vocabulary
		nm = self.l2_normalize_matrix(matrix, matrix_normalized or measure == "euclidean")
		if measure == "cosine":	
			self.faiss_index = faiss.IndexFlatIP(self.dimension)
		elif measure == "euclidean":
			self.faiss_index = faiss.IndexFlatL2(self.dimension)
		print(self.faiss_index.is_trained)
		self.faiss_index.add(nm)
		print(self.faiss_index.ntotal)

	def index_segmented(self, vocabulary, matrix, matrix_normalized = True, nsegments = 100, measure = "cosine"):
		self.index_vocabulary = vocabulary
		nm = self.l2_normalize_matrix(matrix, matrix_normalized or measure == "euclidean")

		print("Initializing the quantizier...")
		if measure == "cosine":	
			self.quantizier = faiss.IndexFlatIP(self.dimension)
		elif measure == "euclidean":
			self.quantizier = faiss.IndexFlatL2(self.dimension)
		
		print("Initializing the segmented index...")
		self.faiss_index = faiss.IndexIVFFlat(self.quantizier, self.dimension, nsegments, faiss.METRIC_INNER_PRODUCT if measure == "cosine" else faiss.METRIC_L2)
		
		print("Segmenting the index...")
		assert not self.faiss_index.is_trained
		self.faiss_index.train(nm)
		assert self.faiss_index.is_trained
		self.faiss_index.add(nm)
		print(self.faiss_index.is_trained)
		print(self.faiss_index.ntotal)


	def search(self, vocabulary, query_matrix, k = None, matrix_normalized = True, measure = "cosine"):	
		self.query_vocabulary = vocabulary
		qm = self.l2_normalize_matrix(query_matrix, matrix_normalized or measure == "euclidean")
		num = k if k else self.faiss_index.ntotal
		self.sims, self.indices = self.faiss_index.search(qm, num)
		return self.sims, self.indices
		