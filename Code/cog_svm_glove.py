import numpy as np
import codecs
import random
import csv
import utils
import re
import dill
import pickle
from collections import defaultdict
from sklearn.externals import joblib
from sklearn import svm, model_selection
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from utils import get_filtered_questions, get_data_for_cognitive_classifiers

np.random.seed(42)

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K' 

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}
X = []
Y_cog = []
Y_know = []

TRAIN_SVM_GLOVE = True
TEST_SVM_GLOVE = True

domain = pickle.load(open('resources/domain_2.pkl',  'rb'))

keywords = set()
for k in domain:
    keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))
    
def lamb1(x):
    return x

def lamb2():
    global gVar
    return gVar

gVar = None
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec['the'])
    
    def _apply_weight(self, w):
        if w in keywords:
            for k in domain:
                if w in domain[k]:
                    break

            return self.word2vec[w] + ((mapping_cog[k] + 1))
        else:
            return self.word2vec[w]

    def fit(self, X, y):
        # not implementing this now
        return self
    
    def transform(self, X, mean=True):
        main_temp = []
        temp = []
        if mean:
            return np.array([
                np.mean([self._apply_weight(w)
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X])
        else:
            for words in X:
                temp = []
                for w in words:
                    if w in self.word2vec: #and w in keywords:
                        temp.append(self.word2vec[w] * self.word2weight[w])
                    else:
                        temp.append(np.zeros(self.dim))
                main_temp.append(temp)
        main_temp = np.array(main_temp)
        return main_temp
               
################ BEGIN LOADING DATA ################

X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers(0.15, 'ada', 0.8)
print('Loaded/Preprocessed data')

vocabulary = {'the'}

for x in X_train + X_test:
    vocabulary = vocabulary.union(set(x))

################ BEGIN TRAINING CODE ################

if TRAIN_SVM_GLOVE:
################ Load Glove w2v only if training is required    #################
    print()
    with open("models/glove.6B.100d.txt", "r", encoding='utf-8') as lines:
        w2v = {}
        for row, line in enumerate(lines):
            try:
                w = line.split()[0]
                vec = np.array(list(map(float, line.split()[1:])))

                if w in vocabulary:
                    w2v[w] = vec
            except:
                continue
            finally:
                print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} GloVe vectors'.format(row + 1))
            
    print('Loaded Glove w2v')

    parameters = {'kernel': ['linear'], 'gamma': [1e-6, 1e-5, 1e-4], 'C': [0.001, 0.05, 0.1, 0.5, 1]}
                 
    gscv = model_selection.GridSearchCV(svm.SVC(), parameters, n_jobs=-1)
    clf = Pipeline([ ('GloVe-Vectorizer', TfidfEmbeddingVectorizer(w2v)), 
                          ('SVC', gscv) ])

    clf.fit(X_train, Y_train)
    print('Fitting done')

    print('Best params:', gscv.best_params_)

    joblib.dump(clf, 'models/glove_svm_model.pkl') 
    print('Saving done')

################ BEGIN TESTING CODE ################
if TEST_SVM_GLOVE:
    if not TRAIN_SVM_GLOVE:
        clf = joblib.load('models/glove_svm_model.pkl')

    Y_true, Y_pred = Y_test, clf.predict(X_test)

    nCorrect = 0
    print('Incorrectly labelled data: ')
    for i in range(len(Y_true)):
        if Y_true[i] != Y_pred[i]:
            print(X_test[i], '| Predicted:', Y_pred[i], '| Actual:', Y_true[i])
        else:
            nCorrect += 1

    print()
    print('Accuracy: {:.3f}%'.format(nCorrect / len(Y_test) * 100))

    print(classification_report(Y_true, Y_pred))

#print(utils.get_glove_vector(['What is horners rule?']))