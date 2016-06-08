from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec, Word2Vec
import numpy as np
from random import shuffle
import logging
import os.path
import sys
import cPickle as pickle
from scipy.spatial.distance import cosine

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.mapping = {}
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    myline = utils.to_unicode(line).split()
                    label = prefix + '_%s' % item_no
                    self.sentences.append(LabeledSentence(myline, [label]))
                    self.mapping[prefix + '_%s' % item_no] = line
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

    def get_sentence(self, key):
    	return self.mapping[key]

sources = {'data/train-neg.txt':'TRAIN_NEG', 'data/train-pos.txt':'TRAIN_POS'}
sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=25, window=10, size=64, workers=8)
model.build_vocab(sentences.to_array())

word_model = Word2Vec.load("model/word_model.mod")
for word in word_model.vocab.keys():
	this_index = model.vocab[word].index
	model.syn0[this_index] = word_model[word]
	model.syn0_lockf[this_index] = 0
assert np.sum(model.syn0_lockf) == len(model.vocab) - len(word_model.vocab)

for epoch in range(10):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())

print model["time"] == word_model["time"] # should be all equal 

model.save("model/doc_model.mod")

################ display mapping results 
def get_similar_doc(word):
	score_dict = {}
	for doc_name in model.docvecs.doctags.keys():
		score_dict[doc_name] = -cosine(model.docvecs[doc_name], model[word])
	sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
	return sorted_list[:5]

print "\n\n\n\n\n"
for doc, score in get_similar_doc("time"):
	print sentences.get_sentence(doc)

print "\n\n\n\n\n"
for doc, score in get_similar_doc("funny"):
	print sentences.get_sentence(doc)

print "\n\n\n\n\n"
for doc, score in get_similar_doc("sexy"):
	print sentences.get_sentence(doc)
