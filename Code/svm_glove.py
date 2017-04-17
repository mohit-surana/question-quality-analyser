import os
import pickle
from collections import defaultdict

import dill
import numpy as np
from sklearn import model_selection, svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from utils import get_data_for_cognitive_classifiers

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

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec['the'])
        
    def fit(self, X, y):
        global gVar
        tfidf = TfidfVectorizer(norm='l2',
                                min_df=1,
                                decode_error="ignore",
                                use_idf=True,
                                smooth_idf=False,
                                sublinear_tf=True,
                                analyzer=lamb1)
        tfidf.fit(X + list(keywords))

        max_idf = max(tfidf.idf_)

        self.word2weight = defaultdict(lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X, mean=True):
        main_temp = []
        temp = []
        if mean:
            return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec and w in keywords] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        else:
            for words in X:
                temp = []
                for w in words:
                    if w in self.word2vec and w in keywords:
                        temp.append(self.word2vec[w] * self.word2weight[w])
                    else:
                        temp.append(np.zeros(self.dim))
                main_temp.append(temp)
        main_temp = np.array(main_temp)
        return main_temp

def train(X_train, Y_train):
# Load Glove w2v only if training is required
    print()
    filename = 'glove.6B.%dd.txt' %100
    
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
                    if((row + 1) % 100000 == 0):
                        print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} GloVe vectors'.format(row + 1))
        
        dill.dump(w2v, open('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0], 'wb'))
    else:
        w2v = dill.load(open('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0], 'rb'))
            
    print('Loaded Glove w2v')

    parameters = {'kernel' : ['poly'],
                  'C': [0.5]}
                 
    gscv = model_selection.GridSearchCV(svm.SVC(decision_function_shape='ovr', verbose=True, class_weight='balanced', probability=True), parameters, n_jobs=-1)
    clf = Pipeline([ ('GloVe-Vectorizer', TfidfEmbeddingVectorizer(w2v)),
                          ('SVC', gscv) ])

    clf.fit(X_train, Y_train)
    print('Fitting done')

    print('Best params:', gscv.best_params_)

    joblib.dump(clf, 'models/SVM/glove_svm_model.pkl')
    print('Saving done')
    return clf
		
if __name__ == '__main__':
    ################ BEGIN LOADING DATA ################

    X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers([0, 0.1, 0.15, 0.2], ['ada', 'os', 'bcl'], 0.8, include_keywords=False)
    print('Loaded/Preprocessed data')

    vocabulary = {'the'}

    ################ BEGIN TRAINING CODE ################

    if TRAIN_SVM_GLOVE:
        clf = train(X_train, Y_train)

    ################ BEGIN TESTING CODE ################
    if TEST_SVM_GLOVE:
        if not TRAIN_SVM_GLOVE:
            clf = joblib.load('models/SVM/glove_svm_model_83.pkl')

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
