import brnn
import csv
import dill
import re
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas
import pickle
import random
import seaborn

from brnn import BiDirectionalRNN, sent_to_glove, clip
from utils import get_filtered_questions, clean_no_stopwords, clean, get_data_for_cognitive_classifiers
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from maxent import features
from svm_glove import TfidfEmbeddingVectorizer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

domain = pickle.load(open('resources/domain.pkl',  'rb'))
domain = { k : set(clean_no_stopwords(' '.join(list(domain[k])), stem=False)) for k in domain.keys() }
domain_names = domain.keys()

keywords = set()
for k in domain:
	keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_cog2 = { v : k for k, v in mapping_cog.items()}

LOAD_MODELS = True

# transformation for BiRNN. This should actually become a part of the RNN for better code maintainability
INPUT_SIZE = 300
NUM_QUESTIONS = 1000
filename = 'glove.6B.%dd.txt' %INPUT_SIZE

if not os.path.exists('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0]):
	print()
	with open('resources/GloVe/' + filename, "r", encoding='utf-8') as lines:
		w2v = {}
		for row, line in enumerate(lines):
			try:
				w = line.split()[0]
				vec = np.array(list(map(float, line.split()[1:])))
				w2v[w] = vec
			except:
				continue
			finally:
				print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} GloVe vectors'.format(row + 1))
	
	dill.dump(w2v, open('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0], 'wb'))
else:
	w2v = dill.load(open('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0], 'rb'))

print('Loaded GloVe model')

if LOAD_MODELS:
	################ MODEL LOADING ##################
	################# MAXENT MODEL #################
	clf_maxent = pickle.load(open('models/MaxEnt/maxent_85.pkl', 'rb'))
	print('Loaded MaxEnt model')
	
	################# SVM-GLOVE MODEL #################
	clf_svm = joblib.load('models/SVM/glove_svm_model.pkl')
	print('Loaded SVM-GloVe model')
	
	################# BiRNN MODEL #################
	clf_brnn = dill.load(open('models/BiRNN/brnn_model.pkl', 'rb'))
	print('Loaded BiRNN model')

######### GET LABEL FOR EXERCISE QUESTIONS #########
X_train1, Y_train1, X_test1, Y_test1 = get_data_for_cognitive_classifiers(threshold=[0.15, 0.25], what_type=['ada', 'bcl', 'os'], split=0.8, include_keywords=False, keep_dup=False, shuffle=False)

X_all_data = list(zip(X_train1 + X_test1, Y_train1 + Y_test1))
random.shuffle(X_all_data)
X1 = [x[0] for x in X_all_data if len(x[0]) > 0]
Y1 = [x[1] for x in X_all_data if len(x[0]) > 0]

ptest_svm = clf_svm.predict(X1)

ptest_brnn = []
for x in sent_to_glove(X1, w2v) :
	ptest_brnn.append(clf_brnn.forward(clip(x)))

ptest_maxent = []
for x in [features(X1[i]) for i in range(len(X1))]:
	ptest_maxent.append(clf_maxent.classify(x))

print('Loaded data for voting system')

X = np.array(list(zip(ptest_maxent, ptest_brnn, ptest_svm)))
Y = np.array(Y1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

###### NEURAL NETWORK BASED VOTING SYSTEM ########
clf = joblib.load('models/cog_ann_voter.pkl')
y_real, y_pred = y_test, clf.predict(x_test)

print('Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred) * 100))
print('MaxEnt Accuracy: {:.2f}%'.format(accuracy_score(y_real, x_test.T[0]) * 100))
print('BiRNN Accuracy: {:.2f}%'.format(accuracy_score(y_real, x_test.T[1]) * 100))
print('SVM-GloVe Accuracy: {:.2f}%'.format(accuracy_score(y_real, x_test.T[2]) * 100))

svm_probs = clf_svm.predict_proba(X_test1)
for i in range(len(svm_probs)):
	probs = svm_probs[i]
	svm_probs[i] = np.exp(probs) / np.sum(np.exp(probs))

svm_probs = np.array(svm_probs)

ptest_maxent = []
for x in [features(X1[i]) for i in range(len(X1))]:
	ptest_maxent.append(clf_maxent.prob_classify(x))

maxent_probs = []
for prob_dist in ptest_maxent:
	pd = prob_dist._prob_dict
	probs = np.array([pd[x] for x in range(6)])
	probs = np.exp(probs) / np.sum(np.exp(probs))
	maxent_probs.append(probs)

maxent_probs = np.array(maxent_probs)

brnn_probs = []
for x in sent_to_glove(X1, w2v):
	probs = clf_brnn.predict_proba(clip(x))
	probs = [x[0] for x in probs]
	probs = np.exp(probs) / np.sum(np.exp(probs))
	brnn_probs.append(probs)

brnn_probs = np.array(brnn_probs)

svm, maxent, brnn = {}, {}, {}

for svm_p, maxent_p, brnn_p, label in zip(svm_probs, maxent_probs, brnn_probs, y_test):
	if(label not in svm):
		svm[label], maxent[label], brnn[label] = list(), list(), list()
	svm[label].append(svm_p[label])
	maxent[label].append(maxent_p[label])
	brnn[label].append(brnn_p[label])


actual = {}
for label in range(6):
	print(label)
	data = pandas.DataFrame({'svm': svm[label], 'maxent': maxent[label], 'brnn': brnn[label], 'actual': [1 for x in svm[label]]})
	seaborn.pairplot(data, vars=['svm', 'maxent', 'brnn'], kind='reg')
	plt.show()
	break
