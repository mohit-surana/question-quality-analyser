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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from utils import get_data_for_cognitive_classifiers, get_glove_vectors, clean_no_stopwords

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

domain = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/domain.pkl'),  'rb'))

keywords = set()
keyword_doc = ['' for i in range(len(mapping_cog))]
for k in domain:
    for word in domain[k]:
        cleaned_word = clean_no_stopwords(word, lemmatize=False, stem=False, as_list=False)
        keywords.add(cleaned_word)
        keyword_doc[mapping_cog[k]] += cleaned_word + '. '

class TfidfEmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.word2weight = None
        self.w2v = w2v
        self.dim = len(w2v['the'])
        
    def fit(self, X, Y):
        tfidf = TfidfVectorizer(norm='l2',
                                min_df=1,
                                decode_error="ignore",
                                use_idf=True,
                                smooth_idf=False,
                                sublinear_tf=True)

        docs_train = ['' for i in range(len(mapping_cog))]

        for x, y in zip(X, Y):
            docs_train[y] += ' '.join(x) + '. '

        for i in range(len(docs_train)):
            docs_train[i] += keyword_doc[i]

        tfidf.fit(docs_train)

        max_idf = max(tfidf.idf_)
        self.max_idf = max_idf

        self.word2weight = defaultdict(lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        keyword_dict = { k : max_idf for k in keywords}

        self.word2weight.update(keyword_dict)

        return self
    
    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] * (self.word2weight[w])
                     for w in words if w in self.w2v] or
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

    X_train, Y_train = get_data_for_cognitive_classifiers(threshold=[0.10, 0.10],
                                                          what_type=['bcl'],
                                                          include_keywords=True,
                                                          keep_dup=False)
    print(len(X_train))

    X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.10],
                                                        what_type=['bcl'],
                                                        what_for='test',
                                                        keep_dup=False)


    print('Loaded/Preprocessed data')

    vocabulary = {'the'}

    if TRAIN:
        parameters = {'kernel' : ['poly'], 'C': [0.7]}
                     
        vec = TfidfEmbeddingVectorizer(w2v)
        gscv = model_selection.GridSearchCV(svm.SVC(decision_function_shape='ovr', verbose=False, class_weight='balanced', probability=True), parameters, n_jobs=-1)
        clf = Pipeline([ ('GloVe-Vectorizer', vec),
                              ('SVC', gscv) ])

        clf.fit(X_train, Y_train)
        print('Fitting done')

        print('Best params:', gscv.best_params_)

        save_svm_model(clf)
        print('Saving done')

    ################ BEGIN TESTING CODE ################
    if TEST:
        clf = load_svm_model('glove_svm_model_bcl.pkl', w2v)

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
        print(confusion_matrix(Y_true, Y_pred))
    #print(utils.get_glove_vector(['What is horners rule?']))
