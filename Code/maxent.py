import pickle

import nltk
import nltk.corpus
from nltk import MaxentClassifier, classify

from utils import clean_no_stopwords, get_data_for_cognitive_classifiers

try:
	domain = pickle.load(open('resources/domain_2.pkl',  'rb'))
except:
	domain = pickle.load(open('resources/domain.pkl',  'rb'))

domain = { k : set(clean_no_stopwords(' '.join(list(domain[k])), stem=False)) for k in domain.keys() } 
inverted_domain = {}
for k in domain:
	for v in domain[k]:
		inverted_domain[v] = k

domain_names = domain.keys()

keywords = set()
for k in domain:
    keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))


mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_cog2 = { v : k for k, v in mapping_cog.items()}
mapping_cog_score = {0 : 0, 1 : 10, 2 : 100, 3 : 1000, 4 : 10000, 5 : 100000}

def features(question):
	features = { 'count({})'.format(d) : sum([1 for w in question if w in domain[d]]) for d in domain_names }

	max_d = ''
	for d in ['Remember', 'Understand', 'Apply', 'Analyse', 'Evaluate', 'Create']:
		if features['count(%s)' %d] > 0:
			max_d = d
	if max_d != '':
		features['class'] = max_d

	for i, word in enumerate(question):
		if word in keywords:
			features['isKeyword(%s)' %word] = True
			features['type(%s)' %word] = inverted_domain[word]
		else:
			features['type({})'.format(word)] = None
	

	#features['keyscore'] = sum([ (mapping_cog_score[mapping_cog[d]]) * features['count({})'.format(d)] for d in domain ])
	
	return features

def check_for_synonyms(word):
    synonyms = set([word])
    for s in nltk.corpus.wordnet.synsets(word):
        synonyms = synonyms.union(set(s.lemma_names()))
    return '@'.join(list(synonyms))

if __name__ == '__main__':
	TRAIN = True
	
	X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.5, 0.6, 0.75, 0.8], what_type=['ada', 'bcl', 'os'], split=0.8, include_keywords=False, keep_dup=False)
	print('Loaded/Preprocessed data')

	X = X_train + X_test
	Y = Y_train + Y_test

	'''
	ctr = { i : ([], []) for i in range(6)}

	for x, y in zip(X, Y):
		ctr[y][0].append(x)
		ctr[y][1].append(y)

	X = []
	Y = []
	for k in ctr:
		X.extend(ctr[k][0])
		Y.extend(ctr[k][1])

	data = list(zip(X, Y))
	random.shuffle(data)
	X = [t[0] for t in data]
	Y = [t[1] for t in data]
	'''

	featuresets = [(features(X[i]), Y[i]) for i in range(len(X))]

	train_percentage = 0.80

	train_set, test_set = featuresets[ : int(len(X) * train_percentage)], featuresets[int(len(X) * train_percentage) : ]

	if TRAIN:
		classifier = MaxentClassifier.train(train_set, max_iter=100)
		classifier.predict_proba = classifier.prob_classify
		pickle.dump(classifier, open('models/MaxEnt/maxent.pkl', 'wb'))

	if not TRAIN:
		classifier  = pickle.load(open('models/MaxEnt/maxent_85.pkl', 'rb'))
	
	pred = []
	actual = [x[1] for x in test_set]
	for t, l in test_set:
		pred.append(classifier.classify(t))

	print(pred)
	print(actual)

	print(classify.accuracy(classifier, test_set))

	'''
	questions = open('datasets/ADA_v2_Exercise_Questions.txt').read().split('\n')

	X_ada = []
	X_ada_orig = []
	for q in questions:
		X_ada.append(clean(q, return_as_list=False, stem=False))
		X_ada_orig.append(q)
	X_ada = get_filtered_questions(X_ada, threshold=0.75, what_type='ada')

	X_ada_features = [features(k.split()) for k in X_ada]

	preds = []
	for t in X_ada_features:
		preds.append(classifier.classify(t))

	with open('datasets/ADA_v2_Exercise_Questions_Labelled.csv', 'w', encoding="utf-8") as csvfile:
	    csvwriter = csv.writer(csvfile)
	    for q, p in zip(X_ada_orig, preds):
	    	csvwriter.writerow([q, mapping_cog2[p]])
	'''
