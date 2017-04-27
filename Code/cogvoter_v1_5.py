import csv
import re
import numpy as np
import pickle
import random
import brnn
import os

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from maxent import features
from svm_glove import TfidfEmbeddingVectorizer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from brnn import BiDirectionalRNN, RNN, relu, relu_prime, sent_to_glove, clip, load_brnn_model
from svm_glove import TfidfEmbeddingVectorizer, foo, load_svm_model
from mnbc import MNBC
from utils import get_filtered_questions, clean_no_stopwords, get_data_for_cognitive_classifiers, get_glove_vectors


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
CUSTOM_GLOVE_SVM = True
CUSTOM_GLOVE_BRNN = False 

savepath = 'glove.%dd%s.pkl' %(VEC_SIZE_SVM, '_custom' if CUSTOM_GLOVE_SVM else '')
svm_w2v = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath), 'rb'))    
    
savepath = 'glove.%dd%s.pkl' %(VEC_SIZE_BRNN, '_custom' if CUSTOM_GLOVE_BRNN else '')
brnn_w2v = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath), 'rb'))

print('Loaded GloVe models')

####################### ONE TIME MODEL LOADING #########################
def get_cog_models(get_ann=True):
    ################# BRNN MODEL #################
    clf_brnn = load_brnn_model('brnn_model_91.pkl', brnn_w2v)
    print('Loaded BiRNN model')
    
    ################# SVM-GLOVE MODEL #################
    clf_svm = load_svm_model('glove_svm_model_81-100d-custom.pkl', svm_w2v)
    print('Loaded SVM-GloVe model')
    
    ################# MNBC MODEL #################
    clf_mnbc = joblib.load(os.path.join(os.path.dirname(__file__), 'models/MNBC/mnbc_89.pkl'))
    print('Loaded MNBC model')

    ################# MLP MODEL #################
    nn = None
    if get_ann:
        nn = joblib.load(os.path.join(os.path.dirname(__file__), 'models/cog_ann_voter.pkl'))
        print('Loaded MLP model')

    return clf_svm, clf_mnbc, clf_brnn, nn

##################### PREDICTION WITH PARAMS ############################
def predict_cog_label(question, models, subject='ADA'):
    clf_svm, clf_mnbc, clf_brnn, nn = models
    question2 = get_filtered_questions(question, threshold=0.5, what_type=subject.lower()) # svm and birnn
    if len(question2) > 0:
        question = question2[0]
    X1 = np.array(question.split()).reshape(1, -1)

    # softmax probabilities
    ptest_svm = clf_svm.predict_proba(X1)
    for i in range(len(ptest_svm)):
        probs = ptest_svm[i]
        ptest_svm[i] = np.exp(probs) / np.sum(np.exp(probs))
    ptest_svm = np.array(ptest_svm)

    ptest_mnbc = clf_mnbc.predict_proba(X1)
    for i in range(len(ptest_mnbc)):
        probs = ptest_mnbc[i]
        ptest_mnbc[i] = np.exp(probs) / np.sum(np.exp(probs))
    ptest_mnbc = np.array(ptest_mnbc)

    ptest_brnn = clf_brnn.predict_proba(X1)
    for i in range(len(ptest_brnn)):
        probs = ptest_brnn[i]
        ptest_brnn[i] = np.exp(probs) / np.sum(np.exp(probs))
    ptest_brnn = np.array(ptest_brnn)

    X = np.hstack((ptest_brnn, ptest_svm, ptest_mnbc)).reshape(1, -1) # concatenating the vectors
    return nn.predict(X)[0], nn.predict_proba(X)[0]


#########################################################################
#                            MAIN BEGINS HERE                           #
#########################################################################
if __name__ == '__main__':
 
    ################ MODEL LOADING ##################
    clf_svm, clf_mnbc, clf_brnn, _ = get_cog_models(get_ann=False)

    ######### GET LABEL FOR EXERCISE QUESTIONS #########
    X_train1, Y_train1, X_test1, Y_test1 = get_data_for_cognitive_classifiers(threshold=[0.15, 0.2, 0.25, 0.3], what_type=['ada', 'os'], split=0.8, include_keywords=True, keep_dup=False)
    X_all_data = list(zip(X_train1 + X_test1, Y_train1 + Y_test1))
    random.shuffle(X_all_data)
    X1 = [x[0] for x in X_all_data if len(x[0]) > 0]
    Y1 = [x[1] for x in X_all_data if len(x[0]) > 0]

    # softmax probabilities
    ptest_svm = clf_svm.predict_proba(X1)
    for i in range(len(ptest_svm)):
        probs = ptest_svm[i]
        ptest_svm[i] = np.exp(probs) / np.sum(np.exp(probs))
    ptest_svm = np.array(ptest_svm)

    ptest_mnbc = clf_mnbc.predict_proba(X1)
    for i in range(len(ptest_mnbc)):
        probs = ptest_mnbc[i]
        ptest_mnbc[i] = np.exp(probs) / np.sum(np.exp(probs))
    ptest_mnbc = np.array(ptest_mnbc)

    ptest_brnn = clf_brnn.predict_proba(X1)
    for i in range(len(ptest_brnn)):
        probs = ptest_brnn[i]
        ptest_brnn[i] = np.exp(probs) / np.sum(np.exp(probs))
    ptest_brnn = np.array(ptest_brnn)

    print('Loaded data for voting system')

    X = np.hstack((ptest_brnn, ptest_svm, ptest_mnbc)) # concatenating the vectors
    Y = np.array(Y1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    ###### NEURAL NETWORK BASED VOTING SYSTEM ########
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(32, 16), batch_size=16, learning_rate='adaptive', learning_rate_init=0.001, verbose=True)
    clf.fit(x_train, y_train)
    print('ANN training completed')
    y_real, y_pred = y_test, clf.predict(x_test)

    joblib.dump(clf, os.path.join(os.path.dirname(__file__), 'models/cog_ann_voter.pkl'))

    print('Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred) * 100))

    y_pred_svm = []
    y_pred_mnbc = []
    y_pred_brnn = []

    for x in x_test:
        y_pred_svm.append(np.argmax(x[:6]))
        y_pred_mnbc.append(np.argmax(x[6:12]))
        y_pred_brnn.append(np.argmax(x[12:]))

    print('SVM-GloVe Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred_svm) * 100))
    print('MNBC Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred_mnbc) * 100))
    print('BiRNN Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred_brnn) * 100))
