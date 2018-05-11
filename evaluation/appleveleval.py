from helpers import io_helper
from embeddings import text_embeddings
from evaluation import simple_stats
from scipy import stats

def evaluate_reps_simlex(embs, simlex_entries, lang = 'en', lang_prefix_first = None, lang_prefix_second = None, lower = False):	
	golds = []
	sims = []
	for entry in simlex_entries:
		w1 = lang_prefix_first + "__" + entry[0].strip() if lang_prefix_first else entry[0].strip()
		w2 = lang_prefix_first + "__" + entry[1].strip() if lang_prefix_first else entry[1].strip()
		if lower:
			w1 = w1.lower()
			w2 = w2.lower()	
		vec1 = embs.get_vector(lang, w1)
		vec2 = embs.get_vector(lang, w2)
		if vec1 is not None and vec2 is not None:
			golds.append(float(entry[2]))
			sims.append(simple_stats.cosine(vec1, vec2))
		if vec1 is None:
			print("Not found: " + w1)
		if vec2 is None:
			print("Not found: " + w2)
	print("Correlation evaluating on " + str(len(golds)) + " pairs")
	spearman = stats.spearmanr(sims, golds)
	pearson = stats.pearsonr(sims, golds)
	return spearman[0], pearson[0]


