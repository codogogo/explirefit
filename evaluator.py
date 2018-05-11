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
parser.add_argument('evaldata', help='A path to the file containing the evaluation dataset, e.g., SimLex-999 (format: word1 \t word2 \t score, one pair per line).')
parser.add_argument('modelpath', help='A path to which to store the trained specialization model.')
args = parser.parse_args()

if not os.path.isfile(args.embs):
	print("Error: File with the pretrained embeddings not found.")
	exit(code = 1)

if not os.path.isfile(args.evaldata):
	print("Error: File with the evaluation dataset not found.")
	exit(code = 1)

if not os.path.isfile(args.modelpath):
	print("Error: Model file not found.")
	exit(code = 1)

embs_path = args.embs
simlex_path = args.evaldata
model_path = args.modelpath

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

# loading simlex and evaluating initial embeddings
simlex_path_en = simlex_path
simlex_entries_en = io_helper.load_csv_lines(simlex_path_en, delimiter = '\t', indices = [0, 1, 3])
simlex_corr_en = appleveleval.evaluate_reps_simlex(t_embeddings, simlex_entries_en, lang = "en", lower = False)
print("Evaluation dataset correlation before specialization: " + str(simlex_corr_en))

# preparing simlex pairs for the computation of the new embeddings with the model
simlex_data = []
for sim_ent in simlex_entries_en:
	if sim_ent[0] in t_embeddings.lang_vocabularies["en"] and sim_ent[1] in t_embeddings.lang_vocabularies["en"]:
		simlex_data.append((t_embeddings.lang_vocabularies["en"][sim_ent[0]], t_embeddings.lang_vocabularies["en"][sim_ent[1]], float(sim_ent[2])))
simlex_data_x1s = [x[0] for x in simlex_data]
simlex_data_x2s = [x[1] for x in simlex_data]
simlex_golds = [x[2] for x in simlex_data]

model = wordpair_model.WordPairModel(embeddings, embedding_size, hidden_layer_sizes, same_mlp = same_encoder, activation = tf.nn.tanh, distance_measure = distance_measure)

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

print("Setting model variables...")
model.set_variable_values(session, vars)

print("Obtaining transformed vectors for evaluation dataset entries...")

first_embs_transformed_simlex = model.mlp1.outputs.eval(session = session, feed_dict = { model.input_w1 : simlex_data_x1s, model.dropout : 1.0, model.mlp1.dropout : 1.0})
second_embs_transformed_simlex = model.mlp2.outputs.eval(session = session, feed_dict = { model.input_w2 : simlex_data_x2s, model.dropout : 1.0, model.mlp2.dropout : 1.0})

simlex_predicted = []
for i in range(len(first_embs_transformed_simlex)):
	simlex_predicted.append(simple_stats.cosine(first_embs_transformed_simlex[i], second_embs_transformed_simlex[i]))

spearman_simlex = stats.spearmanr(simlex_predicted, simlex_golds)
pearson_simlex = stats.pearsonr(simlex_predicted, simlex_golds)

print("Evaluation dataset correlation after specialization: ")
print("Spearman: " + str(spearman_simlex[0]))	
print("Pearson: " + str(pearson_simlex[0]))