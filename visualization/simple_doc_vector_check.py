from gensim.models import Doc2Vec, Word2Vec
import random
import numpy as np
from scipy.spatial.distance import cosine

model = Doc2Vec.load("../model/doc_model.mod")
doc_labels = random.sample(model.docvecs.doctags.keys(), 20)

def get_similar_words(doc):
	score_dict = {}
	for word in model.vocab.keys():
		score_dict[word] = cosine(model[word], doc)
	sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1])
	from nltk.corpus import stopwords
	stopwords = stopwords.words('english')
	return [word for word in sorted_list[:100] if word not in stopwords][:10]

for label in doc_labels:
	doc_vec = model.docvecs[label]
	labels = get_similar_words(doc_vec)
	print "\nLabel %s: %s\n" % (label, str(labels))

