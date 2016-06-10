from minisom import MiniSom
from gensim.models import Doc2Vec, Word2Vec
import random
import numpy as np
from scipy.spatial.distance import cosine

model = Doc2Vec.load("../model/doc_model.mod")
doc_labels = random.sample(model.docvecs.doctags.keys(), 4000)

#### selection 
doc_vecs = []
for label in doc_labels:
	doc_vecs += [model.docvecs[label]]
doc_vecs = np.array(doc_vecs)

####

print "Clustering..."
N_CLUSTERS = 4
som = MiniSom(4, 4, 64, sigma=0.3, learning_rate=0.5)
som.train_random(doc_vecs, 100)
qnt = som.quantization(doc_vecs)

uniques = []
for i in qnt:
	has_it = False
	for elem in uniques:
		if np.array_equal(elem, i):
			has_it = True
	if not has_it:
		uniques += [i]

####

def get_similar_words(doc):
	score_dict = {}
	for word in model.vocab.keys():
		score_dict[word] = cosine(model[word], doc)
	sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1])
	from nltk.corpus import stopwords
	stopwords = stopwords.words('english')
	return [word for word in sorted_list[:100] if word not in stopwords][:10]

for i, cluster_vec in enumerate(uniques):
	count = sum([1 for vec in qnt if np.array_equal(vec, cluster_vec)])
	labels = get_similar_words(cluster_vec)
	print "\nCluster %d: Count %d, %s\n" % (i, count, str(labels))
