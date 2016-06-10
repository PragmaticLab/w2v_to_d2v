from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
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
                    self.mapping[prefix + '_%s' % item_no] = myline
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

    def get_sentence(self, key):
    	return self.mapping[key]

sources = {'../data/test-neg.txt':'TEST_NEG', '../data/test-pos.txt':'TEST_POS', '../data/train-neg.txt':'TRAIN_NEG', '../data/train-pos.txt':'TRAIN_POS'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=2, window=5, size=100, workers=8)

model.build_vocab(sentences.to_array())

for epoch in range(5):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())

########## get most similar word ##########
print "\n\n\n\n\n"
print model.most_similar("time")

score_dict = {}
for word, left_vector in model.vocab.items():
	score_dict[word] = -cosine(model[word], model["time"])
sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
print sorted_list[:5]

########## get most similar document ##########
print "\n\n\n\n\n"

score_dict = {}
for doc_name in model.docvecs.doctags.keys():
	score_dict[doc_name] = -cosine(model.docvecs[doc_name], model["time"])
sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
print sorted_list[:5]
