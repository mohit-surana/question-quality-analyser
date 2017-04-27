import os
import pickle
from collections import defaultdict

import dill
import pickle
import numpy as np
from sklearn import model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from utils import get_data_for_cognitive_classifiers, get_glove_vectors

np.random.seed(42)

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}
X = []
Y_cog = []
Y_know = []

TRAIN = False
USE_CUSTOM_GLOVE_MODELS = False
TEST = True

VEC_SIZE = 100

domain = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/domain_2.pkl'),  'rb'))

keywords = set()
for k in domain:
    keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))
    
def foo(x):
    return x

class TfidfEmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.word2weight = None
        self.w2v = w2v
        self.dim = len(w2v['the'])
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(norm='l2',
                                min_df=1,
                                decode_error="ignore",
                                use_idf=True,
                                smooth_idf=False,
                                sublinear_tf=True,
                                analyzer=foo)
        tfidf.fit(X + list(keywords))

        max_idf = max(tfidf.idf_)

        self.word2weight = defaultdict(lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] * self.word2weight[w]
                     for w in words if w in self.w2v and w in keywords] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def save_svm_model(clf):
    clf.named_steps['GloVe-Vectorizer'].w2v = None
    joblib.dump(clf, os.path.join(os.path.dirname(__file__), 'models/SVM/glove_svm_model.pkl'))

def load_svm_model(model_name, w2v):
    clf = joblib.load(os.path.join(os.path.dirname(__file__), 'models/SVM/' + model_name))
    clf.named_steps['GloVe-Vectorizer'].w2v = w2v

    return clf
	
if __name__ == '__main__':
    ############# GLOVE LOADING CODE ####################
    if USE_CUSTOM_GLOVE_MODELS:
        savepath = 'glove.%dd_custom.pkl' %VEC_SIZE
    else:
        savepath = 'glove.%dd.pkl' %VEC_SIZE

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath)):
        w2v = {}
        if USE_CUSTOM_GLOVE_MODELS:
            print('Loading custom vectors')
            print()
            w2v.update(get_glove_vectors('resources/GloVe/' + 'glove.ADA.%dd.txt' %VEC_SIZE))
            print()
            w2v.update(get_glove_vectors('resources/GloVe/' + 'glove.OS.%dd.txt' %VEC_SIZE))

        print()
        w2v.update(get_glove_vectors('resources/GloVe/' + 'glove.6B.%dd.txt' %VEC_SIZE))

        pickle.dump(w2v, open(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath), 'wb'))
    else:
        w2v = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath), 'rb'))    
    print('Loaded Glove w2v')

    ################ BEGIN LOADING DATA ################
    #X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers([0.2, 0.25, 0.3, 0.35], ['ada', 'os', 'bcl'], 0.8, include_keywords=True)

    X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.2, 0.25, 0.3, 0.35], 
                                                                          what_type=['ada', 'os'], 
                                                                          split=0.8, 
                                                                          include_keywords=False, 
                                                                          keep_dup=False)

    X2_train, Y2_train, X2_test, Y2_test = get_data_for_cognitive_classifiers(threshold=[0.5, 0.75, 1], 
                                                                          what_type=['bcl'], 
                                                                          split=0.8, 
                                                                          include_keywords=False, 
                                                                          keep_dup=False)

    X_train = X_train + X2_train
    Y_train = Y_train + Y2_train
    X_test = X_test + X2_test
    Y_test = Y_test + Y2_test

    X = X_train + X_test
    Y = Y_train + Y_test

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

    print('Loaded/Preprocessed data')

    vocabulary = {'the'}

    if TRAIN:
        parameters = {'kernel' : ['poly'], 'C': [0.5]}
                     
        vec = TfidfEmbeddingVectorizer(w2v)
        gscv = model_selection.GridSearchCV(svm.SVC(decision_function_shape='ovr', verbose=True, class_weight='balanced', probability=True), parameters, n_jobs=-1)
        clf = Pipeline([ ('GloVe-Vectorizer', vec),
                              ('SVC', gscv) ])

        clf.fit(X_train, Y_train)
        print('Fitting done')

        print('Best params:', gscv.best_params_)

        save_svm_model(clf)
        print('Saving done')

    ################ BEGIN TESTING CODE ################
    if TEST:
        clf = load_svm_model('glove_svm_model.pkl', w2v)

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
