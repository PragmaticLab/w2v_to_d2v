from gensim.models import Doc2Vec, Word2Vec
from sklearn.cluster import AgglomerativeClustering
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
N_CLUSTERS = 20
hc = AgglomerativeClustering(affinity="cosine", linkage="average", n_clusters=N_CLUSTERS)
pred_labels = hc.fit_predict(doc_vecs)

#### 

print "Analyzing..."
cluster_vec_dict = {}
cluster_count_dict = {}
for cluster, docvec in zip(pred_labels, doc_vecs):
	if cluster not in cluster_vec_dict:
		cluster_vec_dict[cluster] = docvec
		cluster_count_dict[cluster] = 1
	else:
		cluster_vec_dict[cluster] += docvec
		cluster_count_dict[cluster] += 1

def get_similar_words(doc):
	score_dict = {}
	for word in model.vocab.keys():
		score_dict[word] = cosine(model[word], doc)
	sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1])
	from nltk.corpus import stopwords
	stopwords = stopwords.words('english')
	return [word for word in sorted_list[:100] if word not in stopwords][:10]

for i in range(N_CLUSTERS):
	count = cluster_count_dict[i]
	labels = get_similar_words(cluster_vec_dict[i] / count)
	print "\nCluster %d: Count %d, %s\n" % (i, count, str(labels))
