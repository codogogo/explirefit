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
import argparse
import os

distance_measure = "cosine"
obj_func = "con_mse"

parser = argparse.ArgumentParser(description='Trains a model for semantic specialization of the distributional embedding space.')
parser.add_argument('embs', help='A path to the file containing the pre-trained distributional embeddings. The words in the embedding file need to be sorted by decreasing frequency in the corpus used for training the vectors.')
parser.add_argument('synpath', help='A path to the file containing the synonymy constraints (see the constraint files coupled with this code).')
parser.add_argument('antpath', help='A path to the file containing the antonymy constraints')
parser.add_argument('modelpath', help='A path to which to store the trained specialization model.')
args = parser.parse_args()

embs_path = args.embs
synonyms_path = args.synpath 
antonyms_path = args.antpath
model_path = args.modelpath

if not os.path.isfile(args.embs):
	print("Error: File with the pretrained embeddings not found.")
	exit(code = 1)

if not os.path.isfile(args.synpath):
	print("Error: File with the synonym constraints not found.")
	exit(code = 1)

if not os.path.isfile(args.antpath):
	print("Error: File with the antonym constraints not found.")
	exit(code = 1)

if args.modelpath is not None and not os.path.isdir(os.path.dirname(args.modelpath)) and not os.path.dirname(args.modelpath) == "":
	print("Error: Directory of the desired model output path not found.")
	exit(code = 1)

# loading/merging word embeddings
print("Loading pre-trained embeddings...")
t_embeddings = text_embeddings.Embeddings()
t_embeddings.load_embeddings(embs_path, 200000, language = 'en', print_loading = True, skip_first_line = False)
t_embeddings.inverse_vocabularies()
vocabulary_size = len(t_embeddings.lang_vocabularies["en"])
embeddings = t_embeddings.lang_embeddings["en"].astype(np.float64)
embedding_size = t_embeddings.emb_sizes["en"]

# data loading and preprocessing
synonym_pairs = data_shaper.prep_word_tuples([[x[0].split('_')[1], x[1].split('_')[1]] for x in io_helper.load_csv_lines(synonyms_path, delimiter = ' ')], t_embeddings, "en")
antonym_pairs = data_shaper.prep_word_tuples([[x[0].split('_')[1], x[1].split('_')[1]] for x in io_helper.load_csv_lines(antonyms_path, delimiter = ' ')], t_embeddings, "en")
all_pairs = []
all_pairs.extend(synonym_pairs)
all_pairs.extend(antonym_pairs)
print("Num syn pairs: " + str(len(synonym_pairs)))
print("Num ant pairs: " + str(len(antonym_pairs)))

syn_constraints_dict = {(str(i1) + ":" + str(i2) if i1 < i2 else str(i2) + ":" + str(i1)) : 0 for (i1, i2) in synonym_pairs}
ant_constraints_dict = {(str(i1) + ":" + str(i2) if i1 < i2 else str(i2) + ":" + str(i1)) : 0 for (i1, i2) in antonym_pairs}

# Faiss wrapper for quick comparison of vectors, initially we index the starting distributional vectors
print("Building FAISS index for fast retrieval of most similar vectors...")
faisser = faiss_sts.Faiss(embedding_size)
faisser.index(None, t_embeddings.lang_embeddings["en"], matrix_normalized = False, measure = distance_measure)

###
### Data preparation, contrastive
###

print("Preparing contrastive micro-batches for training...")
queries = []
query_word_indices = {}
faiss_mapper = {}

for embind_first, embind_second in all_pairs:
	if embind_first not in query_word_indices:	
		queries.append(t_embeddings.lang_embeddings["en"][embind_first])
		query_word_indices[embind_first] = len(faiss_mapper) 
		faiss_mapper[embind_first] = len(faiss_mapper) 
	if embind_second not in query_word_indices:	
		queries.append(t_embeddings.lang_embeddings["en"][embind_second])
		query_word_indices[embind_second] = len(faiss_mapper) 
		faiss_mapper[embind_second] = len(faiss_mapper) 

print("Performing faiss search...")
print("Total queries: " + str(len(queries)))
num_most_sim = 50
sims, indices = faisser.search(None, np.array(queries).astype('float32'), matrix_normalized = False, measure = distance_measure, k = num_most_sim)
			
k_neg = 4
sim_start_index = 1
micro_batches_syn = []
micro_batches_ant = []

num_expelled = 0
print("Preparing synonym micro-batches...")
for i in range(len(all_pairs)):
	if i % 1000 == 0: 
		print("Preparing micro batch: " + str(i+1) + " of " + str(len(all_pairs)))
	if i == len(synonym_pairs):
		print("Preparing antonym micro-batches...")

	embind_first = all_pairs[i][0] 
	embind_second = all_pairs[i][1]
 
	first_word = t_embeddings.get_word_from_index(embind_first, 'en').strip()
	second_word = t_embeddings.get_word_from_index(embind_second, 'en').strip()

	if len(first_word) < 3 or len(second_word) < 3:
		num_expelled += 1
		continue

	negatives_first = []
	most_similar_first = indices[faiss_mapper[embind_first]][sim_start_index:]
	while len(negatives_first) < k_neg:
		j = random.randint(0, num_most_sim - sim_start_index - 1)		
		candidate_word = t_embeddings.get_word_from_index(most_similar_first[j], 'en')
		if not candidate_word:
			continue
			
		if (candidate_word.lower() in second_word.lower()) or (second_word.lower() in candidate_word.lower()) or (candidate_word.lower() in first_word.lower()) or (first_word.lower() in candidate_word.lower()) or most_similar_first[j] in negatives_first:
			continue
		else:
			negatives_first.append(most_similar_first[j])
			
	negatives_second = []	
	most_similar_second = indices[faiss_mapper[embind_second]][sim_start_index:]
	while len(negatives_second) < k_neg:
		j = random.randint(0, num_most_sim - sim_start_index - 1)
		candidate_word = t_embeddings.get_word_from_index(most_similar_second[j], 'en')
		if not candidate_word:
			continue
		if (candidate_word.lower() in second_word.lower()) or (second_word.lower() in candidate_word.lower()) or (candidate_word.lower() in first_word.lower()) or (first_word.lower() in candidate_word.lower()) or most_similar_second[j] in negatives_second:
			continue
		else:
			negatives_second.append(most_similar_second[j])
	
	micro_batch = []
	
	micro_batch.append((embind_first, embind_second, 0.0 if (i < len(synonym_pairs)) else 2.0))

	first_emb = t_embeddings.lang_embeddings['en'][embind_first]
	second_emb = t_embeddings.lang_embeddings['en'][embind_second]

	for ind in negatives_first:
		ind_str1 = (str(embind_first) + ":" + str(ind) if embind_first < ind else str(ind) + ":" + str(embind_first))
		if ind_str1 in syn_constraints_dict:
			dist = 0.0
		elif ind_str1 in ant_constraints_dict:
			dist = 2.0
		else:
			ind_emb = t_embeddings.lang_embeddings['en'][ind]
			dist = 1.0 - simple_stats.cosine(first_emb, ind_emb)
		micro_batch.append((embind_first, ind, dist))
	for ind in negatives_second:
		ind_str1 = (str(embind_second) + ":" + str(ind) if embind_second < ind else str(ind) + ":" + str(embind_second))
		if ind_str1 in syn_constraints_dict:
			dist = 0.0
		elif ind_str1 in ant_constraints_dict:
			dist = 2.0
		else:
			ind_emb = t_embeddings.lang_embeddings['en'][ind]
			dist = 1.0 - simple_stats.cosine(second_emb, ind_emb)
		micro_batch.append((embind_second, ind, dist))

	if i < len(synonym_pairs):
		micro_batches_syn.append(micro_batch)
	else:
		micro_batches_ant.append(micro_batch) 

print("Num micro-batches synonyms: " + str(len(micro_batches_syn)))
print("Num micro-batches antonyms: " + str(len(micro_batches_ant)))

print("Preparing training data (and train/dev split)...")
data = []
data.extend(micro_batches_syn)
data.extend(micro_batches_ant)
random.shuffle(data)

microbatch_size = len(data[0])

batch_size = 100
num_batches = (len(data) // batch_size) + 1

train_ratio = 0.99
train_set = data[ : int(train_ratio*num_batches*batch_size)]
dev_set = data[int(train_ratio*num_batches*batch_size) : ]

print("Train set size: " + str(len(train_set)))
print("Dev set size: " + str(len(dev_set)))

# parameters for grid searching 
num_layers = [5]
hidden_layer_sizes = [1000]
same_mlps = [True]
dist_regs = [0.3]
l2_reg_factors = [0.0001]
learning_rates = [1e-4]
dropouts = [1.0]
configs = itertools.product(num_layers, hidden_layer_sizes, same_mlps, dist_regs, l2_reg_factors, learning_rates, dropouts)
l2 = (obj_func == "con_mse")
print("L2 parameters is " + str(l2))


def prep_model_config(configuration):
	nl, hl_size, same_encoder, dist_reg_fac, l2_reg_fac, lr, drp = configuration 
	conf_str = "num-layers-" + str(nl) + "_hidden_layer_size-" + str(hl_size) + "_same-enc" + str(same_encoder) + "_dist-reg-" + str(dist_reg_fac) + "_l2-reg-" + str(l2_reg_fac) + "_learning-rate-" + str(lr) + "_dropout-" + str(drp)
	print("Running configuration: " + conf_str)
	
	encoder_layers = [hl_size] * (nl - 1) + [embedding_size]
	model = wordpair_model.WordPairModel(embeddings, embedding_size, encoder_layers, same_mlp = same_encoder, activation = tf.nn.tanh, distance_measure = distance_measure)
	if obj_func == "con_mm" or obj_func == "con_mse":
		print("Defining " + obj_func + "objective")
		model.define_optimization(loss_functions.contrastive_loss_exact, dist_reg_fac, l2_reg_fac, lr, loss_function_params = (microbatch_size, batch_size, l2))
	elif obj_func == "simple_mse":
		print("Defining SIMPLE MSE objective")
		model.define_optimization(loss_functions.mse_loss, dist_reg_fac, l2_reg_fac, lr)

	session = tf.InteractiveSession()
	session.run(tf.global_variables_initializer())

	return model, conf_str, session

def build_feed_dict_func(model, data, config = None, predict = False):	
	x1s_flat = []
	x2s_flat = []
	ys_flat = []

	for i in range(len(data)):
		x1s, x2s, ys = zip(*data[i])
		x1s_flat.extend(x1s)
		x2s_flat.extend(x2s)
		ys_flat.extend(ys)
	
	drp = config[-1]			
	fd = model.get_feed_dict(x1s_flat, x2s_flat, ys_flat, 1.0 if predict else drp)
	return fd, ys

# training parameters
max_num_epochs = 1000
num_evals_not_better_end = 30
eval_each_num_batches = 100
shuffle_data = False
	
simp_trainer = trainer.SimpleTrainer(None, None, build_feed_dict_func, None, configuration_func = prep_model_config, additional_results_func = None, model_serialization_path = model_path)
results = simp_trainer.grid_search(configs, train_set, dev_set, batch_size, max_num_epochs, num_devs_not_better_end = num_evals_not_better_end, batch_dev_perf = eval_each_num_batches, print_batch_losses = False, dev_score_maximize = False, print_training = True, shuffle_data = shuffle_data) 
print("Training successfully completed.")