import numpy as np
import tensorflow as tf
from embeddings import text_embeddings
from evaluation import appleveleval
from models import wordpair_model
from helpers import io_helper
from helpers import data_shaper
import random
import itertools
from ml import loss_functions
from ml import trainer
from evaluation import simple_stats
from scipy import stats
from evaluation import standard
from sys import stdin
import sys
from sts import faiss_sts
import os
import argparse

parser = argparse.ArgumentParser(description='Evaluates a specialization model on an evaluation dataset.')
parser.add_argument('embs', help='A path to the file containing the pre-trained (i.e., not specialized) distributional embeddings. The words in the embedding file need to be sorted by decreasing frequency in the corpus used for training the vectors.')
parser.add_argument('modelpath', help='A path to which to store the trained specialization model.')
parser.add_argument('outputpath', help='A filepath to which to write the specialized embeddings.')

args = parser.parse_args()

if not os.path.isfile(args.embs):
	print("Error: File with the pretrained embeddings not found.")
	exit(code = 1)

if args.outputpath is not None and not os.path.isdir(os.path.dirname(args.modelpath)) and not os.path.dirname(args.modelpath) == "":
	print("Error: Directory of the desired output path not found.")
	exit(code = 1)

if not os.path.isfile(args.modelpath):
	print("Error: Model file not found.")
	exit(code = 1)

embs_path = args.embs
model_path = args.modelpath
converted_embs_path = args.outputpath

# deserializing the model 

hyps, vars = io_helper.deserialize(model_path)
print(hyps)
same_encoder, hidden_layer_sizes, distance_measure = hyps

# loading/merging word embeddings
t_embeddings = text_embeddings.Embeddings()
t_embeddings.load_embeddings(embs_path, 200000, language = 'en', print_loading = True, skip_first_line = True)
t_embeddings.inverse_vocabularies()
vocabulary_size = len(t_embeddings.lang_vocabularies["en"])
embeddings = t_embeddings.lang_embeddings["en"].astype(np.float64)
embedding_size = t_embeddings.emb_sizes["en"]

model = wordpair_model.WordPairModel(embeddings, embedding_size, hidden_layer_sizes, same_mlp = same_encoder, activation = tf.nn.tanh, distance_measure = distance_measure)

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

print("Setting model variables...")
model.set_variable_values(session, vars)

print("Converting the input embedding space...")
batch_convert_size = 1000
num_convert_batches = (len(embeddings) // batch_convert_size) + 1
for i in range(num_convert_batches):
	print("Converting embeddings " + str(i * batch_convert_size))
	emb_to_transform = range(i * batch_convert_size, (i + 1) * batch_convert_size) if (i < (num_convert_batches - 1)) else range(i * batch_convert_size, len(embeddings))
	transformed_batch = model.mlp1.outputs.eval(session = session, feed_dict = { model.input_w1 : emb_to_transform, model.dropout : 1.0, model.mlp1.dropout : 1.0})
	if i == 0:
		new_vecs = transformed_batch
	else:
		new_vecs = np.concatenate((new_vecs, transformed_batch), axis = 0)

print("Translation complete.")
print("Len new vecs: " + str(len(new_vecs)))

new_t_embs = text_embeddings.Embeddings()
new_t_embs.lang_vocabularies['new'] = t_embeddings.lang_vocabularies['en']
new_t_embs.lang_embeddings['new'] = new_vecs

io_helper.store_embeddings(converted_embs_path, new_t_embs, 'new', print_progress = True)
print("I'm all done here, ciao bella!")