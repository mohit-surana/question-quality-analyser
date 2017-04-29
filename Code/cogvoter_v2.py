import csv
import re
import numpy as np
import pickle
import random
import brnn
import os
import operator

from brnn import BiDirectionalRNN, RNN, relu, relu_prime, sent_to_glove, clip, load_brnn_model
from svm_glove import TfidfEmbeddingVectorizer, foo, load_svm_model
from mnbc import MNBC
from utils import get_filtered_questions, clean_no_stopwords, clean, get_data_for_cognitive_classifiers, get_glove_vectors
from sklearn import model_selection, svm
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

domain = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/domain.pkl'),  'rb'))
domain = { k : set(clean_no_stopwords(' '.join(list(domain[k])), stem=False)) for k in domain.keys() }
domain_names = domain.keys()

keywords = set()
for k in domain:
    keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_cog2 = { v : k for k, v in mapping_cog.items()}

# transformation for BiRNN. This should actually become a part of the RNN for better code maintainability
VEC_SIZE_SVM = 100
VEC_SIZE_BRNN = 300
NUM_QUESTIONS = 1000
NUM_CLASSES = 6
CUSTOM_GLOVE_SVM = False
CUSTOM_GLOVE_BRNN = False 

savepath = 'glove.%dd%s.pkl' %(VEC_SIZE_SVM, '_custom' if CUSTOM_GLOVE_SVM else '')
svm_w2v = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath), 'rb'))    
    
savepath = 'glove.%dd%s.pkl' %(VEC_SIZE_BRNN, '_custom' if CUSTOM_GLOVE_BRNN else '')
brnn_w2v = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath), 'rb'))

print('Loaded GloVe models')

#################### ENSEMBLE CLASSIFIER ######################
###### Based on code from http://sebastianraschka.com/Articles/2014_ensemble_classifier.html #######
class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for scikit-learn estimators.

    Parameters
    ----------

    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.

    """
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        pass

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """
        y_pred = [] 
        for i, clf in enumerate(self.clfs):
            y_pred.append(clf.predict(X))

        self.classes_ = np.asarray(y_pred)
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):

        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = []
        for i, clf in enumerate(self.clfs):
            self.probas_.append(clf.predict_proba(X))

        self.probas_ = np.array(self.probas_)

        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg



#########################################################################
#                            MAIN BEGINS HERE                           #
#########################################################################
if __name__ == '__main__':
    ######### GET DATA FOR TRAIN/TEST #########
    X_data = []
    Y_data = []

    X_train, Y_train = get_data_for_cognitive_classifiers(threshold=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
                                                            what_type=['ada', 'os', 'bcl'],
                                                            include_keywords=True,
                                                            keep_dup=False)

    X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.25],
                                                        what_type=['ada', 'os', 'bcl'],
                                                        what_for='test',
                                                        include_keywords=False,
                                                        keep_dup=False)

    ################# BRNN MODEL #################
    clf_brnn = load_brnn_model('brnn_model.pkl', brnn_w2v)
    print('Loaded BiRNN model')
    
    ################# SVM-GLOVE MODEL #################
    clf_svm = load_svm_model('glove_svm_model.pkl', svm_w2v)
    print('Loaded SVM-GloVe model')
    
    ################# MNBC MODEL #################
    clf_mnbc = joblib.load(os.path.join(os.path.dirname(__file__), 'models/MNBC/mnbc.pkl'))
    print('Loaded MNBC model')

    eclf = EnsembleClassifier(clfs=[clf_brnn, clf_svm, clf_mnbc], weights=None)

    Y_real, Y_pred = Y_test, eclf.predict(X_test)
    score = accuracy_score(Y_real, Y_pred) * 100
    print('Accuracy: {:.2f}%'.format(accuracy_score(Y_real, Y_pred) * 100))

