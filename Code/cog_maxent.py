import pickle
import nltk
import nltk.corpus
from nltk.classify import apply_features
from nltk import MaxentClassifier, classify
import codecs
import csv
import random
import re

import utils
from utils import clean_no_stopwords, get_data_for_cognitive_classifiers

try:
	domain = pickle.load(open('resources/domain_2.pkl',  'rb'))
except:
	domain = pickle.load(open('resources/domain.pkl',  'rb'))

domain = { k : set(clean_no_stopwords(' '.join(list(domain[k])))) for k in domain.keys() } 

domain_names = domain.keys()

keywords = set()
for k in domain:
    keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))


mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_cog_score = {0 : 0, 1 : 10, 2 : 100, 3 : 1000, 4 : 10000, 5 : 100000}

def features(question):
	features = { 'count({})'.format(d.upper()) : sum([1 for w in question if w in domain[d]]) for d in domain_names }

	features['count(CONTEXT)'] = 0
	features['count(KEYWORD)'] = 0
	for i, word in enumerate(question):
		if word in keywords:
			features['isKeyword({})'.format(word)] = True
			for d in domain:
				if word in domain[d]:
					break

			features['type({})'.format(word)] = d.upper()
			features['count(KEYWORD)'] += 1
		else:
			features['isKeyword({})'.format(word)] = False
			features['type({})'.format(word)] = 'CONTEXT'
			features['count(CONTEXT)'] += 1

		features['count({})'.format(word)] = 1 if 'count({})'.format(word) not in features else (features['count({})'.format(word)] + 1)

	features['keyscore'] = sum([ (mapping_cog_score[mapping_cog[d]]) * features['count({})'.format(d.upper())] for d in domain ])
	
	return features

def check_for_synonyms(word):
    synonyms = set([word])
    for s in nltk.corpus.wordnet.synsets(word):
        synonyms = synonyms.union(set(s.lemma_names()))
    return '@'.join(list(synonyms))

X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.6, 0.7, 0.75, 0.8], what_type=['ada', 'bcl', 'os'], split=0.8, include_keywords=False, keep_dup=True)
print('Loaded/Preprocessed data')

X = X_train + X_test
Y = Y_train + Y_test
featuresets = [(features(X[i]), Y[i]) for i in range(len(X))]

train_percentage = 0.80

train_set, test_set = featuresets[ : int(len(X) * train_percentage)], featuresets[int(len(X) * train_percentage) : ]

classifier = MaxentClassifier.train(train_set, max_iter=15)
print(classify.accuracy(classifier, test_set))

pred = []
actual = [x[1] for x in test_set]
for t, l in test_set:
	pred.append(classifier.classify(t))

#print(pred)
#print(actual)

pickle.dump(classifier, open('models/cog_maxent.pkl', 'wb'))
