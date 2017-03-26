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
from utils import clean_no_stopwords

domain = pickle.load(open('resources/domain_2.pkl',  'rb'))
domain = { k : set(clean_no_stopwords(' '.join(list(domain[k])))) for k in domain.keys() } 

domain_names = domain.keys()

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}

def features(question):
	features = { 'count({})'.format(d) : sum([1 for w in question if w in domain[d]]) for d in domain_names }
	
	# buggy synonym code
	'''
	for word in question:
		if '@' in word:
			synWords = word.split('@')
			flag = 0
			for w in synWords:
				for d in domain:
					if w in domain[d]:
						features['count({})'.format(d)] += 1
						flag = 1
						break
				if flag == 1:
					break
		else:
			for d in domain:
				if word in domain[d]:
					features['count({})'.format(d)] += 1
					break
	'''
	for label in domain:
		for word in domain[label]:
			if question.count(word):
				features[word] = question.count(word)

	return features

def check_for_synonyms(word):
    synonyms = set([word])
    for s in nltk.corpus.wordnet.synsets(word):
        synonyms = synonyms.union(set(s.lemma_names()))

    return '@'.join(list(synonyms))

X = []
Xbcl = []
Y_cog = []
Ybcl_cog = []
with open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile.read().splitlines()[1:])
    for row in csvreader:
        sentence, label_cog, label_know = row
        m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
        sentence = m.groups()[2]
        label_cog = label_cog.split('/')[0]
        X.append(sentence)
        Y_cog.append(mapping_cog[label_cog])

with open('datasets/BCLs_Question_Dataset.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile.read().splitlines())
    for row in csvreader:
        sentence, label_cog = row
        Xbcl.append(sentence)
        Ybcl_cog.append(mapping_cog[label_cog])

dataset = list(zip(Xbcl, Ybcl_cog))
random.shuffle(dataset)
Xbcl = [x[0] for x in dataset]
Ybcl_cog = [x[1] for x in dataset]

# clean and get synonyms
for i in range(len(X)):
	words = clean_no_stopwords(X[i])
	for j, w in enumerate(words):
		words[j] = check_for_synonyms(w)
	X[i] = words

for i in range(len(Xbcl)):
	words = clean_no_stopwords(Xbcl[i])
	'''
	for j, w in enumerate(words):
		words[j] = check_for_synonyms(w)
	'''
	Xbcl[i] = words

featuresets = [(features(X[i]), Y_cog[i]) for i in range(len(X))]
featuresets_bcl = [(features(Xbcl[i]), Ybcl_cog[i]) for i in range(len(Xbcl))]

test_percentage = 0.8

train_set, test_set = featuresets[ : int(len(X) * test_percentage)], featuresets[int(len(X) * test_percentage) : ]
train_set_bcl = featuresets_bcl

train_set = train_set + train_set_bcl
classifier = MaxentClassifier.train(train_set, max_iter=30)
print(classify.accuracy(classifier, test_set))

pred = []
actual = [x[1] for x in test_set]
for t, l in test_set:
	pred.append(classifier.classify(t))

print(pred)
print(actual)
