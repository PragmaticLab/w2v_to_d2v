from gensim.models import Doc2Vec, Word2Vec
import numpy as np

model1 = Doc2Vec.load("model/doc_model1.mod")
model2 = Doc2Vec.load("model/doc_model.mod")

total = 0.0
for doc_name in model1.docvecs.doctags.keys():
	total += np.sum(np.absolute(model1.docvecs[doc_name] - model2.docvecs[doc_name])) / 64

print "avg dimensional difference: %f" % (total / len(model1.docvecs.doctags))