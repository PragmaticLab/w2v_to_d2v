from gensim.models import Word2Vec
import sys

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_corpus(location):
	f = open(location)
	lines = f.read().split('\n')
	corpus = [line.split() for line in lines]
	return corpus

corpus = get_corpus("data/train-pos.txt") + get_corpus("data/train-neg.txt")

model = Word2Vec(corpus, size=64, alpha=0.015, window=10, min_count=25, workers=8, sg=1, hs=1, negative=0, sample=1e-3, iter=10)

model.save("model/word_model.mod")
