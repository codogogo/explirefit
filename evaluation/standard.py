import numpy as np
from evaluation import confusion_matrix

def accuracy(golds, preds, labels = None):
	golds_bin = [(g > 0) for g in golds]
	preds_bin = [(p > 0) for p in preds]
	correct = len([i for i in range(len(golds)) if golds_bin[i] == preds_bin[i]])
	acc = 1.0 * correct / len(golds);
	return acc	

def get_accuracy_seqlab(golds, preds):
	correct = 0.0
	total = 0.0
	for i in range(len(golds)):
		for j in range(len(golds[i])):
			if np.count_nonzero(golds[i][j]) > 0:
				total += 1.0
				lgold = np.argmax(golds[i][j])
				lpred = np.argmax(preds[i][j])
				if lgold == lpred:
					correct += 1.0
	acc = correct / total;
	return acc

def get_performance_seqlab(golds, preds, labels, print = True, print_per_class = True):
	golds_flat = []
	preds_flat = []
	for i in range(len(golds)):
		for j in range(len(golds[i])):
			if np.count_nonzero(golds[i][j]) > 0:
				golds_flat.append(golds[i][j])
				preds_flat.append(preds[i][j])
			else:
				break
	confmat = confusion_matrix.ConfusionMatrix(labels, preds_flat, golds_flat, one_hot_encoding = True)
	confmat.print_results()
	return confmat.accuracy


			
	