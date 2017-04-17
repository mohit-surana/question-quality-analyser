import csv
import dill
import re
import nltk
import numpy as np
import pickle
import random
import brnn
import os
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
CREATE_CSV_FILE = False
TRAIN = False
TEST = False


def predict_cog_label(question):
    X1 = [question]
    # softmax probabilities
    ptest_svm = clf_svm.predict_proba(X1)
    for i in range(len(ptest_svm)):
        probs = ptest_svm[i]
        ptest_svm[i] = np.exp(probs) / np.sum(np.exp(probs))

    ptest_svm = np.array(ptest_svm)

    ptest_maxent = []
    for x in [features(X1[i]) for i in range(len(X1))]:
        p_dict = clf_maxent.prob_classify(x)._prob_dict
        probs = np.array([p_dict[x] for x in range(6)])
        probs = np.exp(probs) / np.sum(np.exp(probs))
        ptest_maxent.append(probs)

    ptest_brnn = []
    for x in sent_to_glove(X1, w2v):
        probs = clf_brnn.predict_proba(clip(x))
        probs = [x[0] for x in probs]
        probs = np.exp(probs) / np.sum(np.exp(probs))
        ptest_brnn.append(probs)

    ptest_brnn = np.array(ptest_brnn)

    print('Loaded question for voting system')
    X = np.hstack((ptest_svm, ptest_maxent, ptest_brnn)) # concatenating the vectors
    print('data is:', X)
    print('level', nn.predict(X)[0])
    print('prob:', nn.predict_proba(X)[0])
    return nn.predict(X)[0], nn.predict_proba(X)[0]


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
                if((row + 1) % 100000 == 0):
                    print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} GloVe vectors'.format(row + 1))
    
    dill.dump(w2v, open('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0], 'wb'))
else:
    w2v = dill.load(open('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0], 'rb'))

print('Loaded GloVe model')

if LOAD_MODELS:
    ################ MODEL LOADING ##################
    ################# MAXENT MODEL #################
    clf_maxent = pickle.load(open('models/MaxEnt/maxent_76.pkl', 'rb'))
    print('Loaded MaxEnt model')
    
    ################# SVM-GLOVE MODEL #################
    clf_svm = joblib.load('models/SVM/glove_svm_model_83.pkl')
    print('Loaded SVM-GloVe model')
    
    ################# BiRNN MODEL #################
    clf_brnn = dill.load(open('models/BiRNN/brnn_model_6B-300_71.pkl', 'rb'))
    print('Loaded BiRNN model')

    ################# MLP MODEL #################
    nn = joblib.load('models/cog_ann_voter_87.pkl')
    print('Loaded MLP model')
if CREATE_CSV_FILE:
    ################# LOADING SO[ADA] questions #################
    ADA_questions = []
    ADA_questions_cleaned = []
    with open('datasets/ADA_SO_Questions.csv', 'r', encoding='utf-8') as csvfile:
        print()
        csvreader = csv.reader(csvfile)
        for i, row in enumerate(csvreader):
            if i == 0 or len(row) == 0:
                continue
            _, sentence, _ = row
            clean_sentence = clean(sentence, return_as_list=False, stem=False)
            if len(set(clean_sentence.split()).intersection(keywords)) and '?' in sentence:
                ADA_questions.append(sentence)
                ADA_questions_cleaned.append(clean_sentence)
                print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} ADA questions'.format(len(ADA_questions)))
                if len(ADA_questions) == NUM_QUESTIONS:
                    break

    ADA_questions_filtered = get_filtered_questions(ADA_questions_cleaned, threshold=0.25, what_type='ada')
    ADA_questions_filtered_for_maxent = get_filtered_questions(ADA_questions_cleaned, threshold=0.75, what_type='ada')

    t_ADA = list(zip(ADA_questions, ADA_questions_cleaned, ADA_questions_filtered_for_maxent, ADA_questions_filtered))
    random.shuffle(t_ADA)
    ADA_questions = [t[0] for t in t_ADA if t[-1].strip() != '']
    ADA_questions_cleaned = [t[1] for t in t_ADA if t[-1].strip() != '']
    ADA_questions_filtered_for_maxent = [t[2] for t in t_ADA if t[-1].strip() != '']
    ADA_questions_filtered = [t[3] for t in t_ADA if t[-1].strip() != '']

    ################# LOADING SO[OS] questions #################
    OS_questions = []
    OS_questions_cleaned = []
    with open('datasets/OS_SO_Questions.csv', 'r', encoding='utf-8') as csvfile:
        print()
        csvreader = csv.reader(csvfile)
        for i, row in enumerate(csvreader):
            if i == 0 or len(row) == 0:
                continue
            _, sentence, _ = row
            clean_sentence = clean(sentence, return_as_list=False, stem=False)
            if len(set(clean_sentence.split()).intersection(keywords)) and '?' in sentence:
                OS_questions.append(sentence)
                OS_questions_cleaned.append(clean_sentence)
                print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} OS questions'.format(len(OS_questions)))

                if len(OS_questions) == NUM_QUESTIONS:
                    break

    ################ MODEL PREDICTIONS ##################
    ################# MAXENT MODEL #################
    pred_maxent = []
    for x in X_data_featureset:
        pred_maxent.append(clf_maxent.classify(x))
    print('MaxEnt classification complete')
    
    ################# SVM-GLOVE MODEL #################
    pred_svm = clf_svm.predict([x.split() for x in  X_data])
    print('SVM-GloVe classification complete')
    
    ################# BiRNN MODEL #################
    pred_brnn = []
    for x in X_data_glove:
        pred_brnn.append(clf_brnn.forward(clip(x)))
    print('BiRNN classification complete')

    OS_questions_filtered = get_filtered_questions(OS_questions_cleaned, threshold=0.25, what_type='os')
    OS_questions_filtered_for_maxent = get_filtered_questions(OS_questions_cleaned, threshold=0.75, what_type='os')
    t_OS = list(zip(OS_questions, OS_questions_cleaned, OS_questions_filtered_for_maxent, OS_questions_filtered))
    random.shuffle(t_OS)
    OS_questions = [t[0] for t in t_OS if t[-1].strip() != '']
    OS_questions_cleaned = [t[1] for t in t_OS if t[-1].strip() != '']
    OS_questions_filtered_for_maxent = [t[2] for t in t_OS if t[-1].strip() != '']
    OS_questions_filtered = [t[3] for t in t_OS if t[-1].strip() != '']

    X_data = ADA_questions_filtered + OS_questions_filtered
    X_data_for_maxent = ADA_questions_filtered_for_maxent + OS_questions_filtered_for_maxent
    X_data_featureset = [features(X_data_for_maxent[i].split()) for i in range(len(X_data_for_maxent))]
    X_data_glove = sent_to_glove(X_data, w2v)
    
    ################# DUMPING OUTPUT TO CSV #################

    with open('datasets/SO_Questions_Cog_Prediction.csv', 'w', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Question', 'Cog(MaxEnt)', 'Cog(BiRNN)', 'Cog(SVM-GloVe)'])
        for q, p_maxent, p_brnn, p_svm in zip(ADA_questions + OS_questions, pred_maxent, pred_brnn, pred_svm):
            csvwriter.writerow([q, mapping_cog2[p_maxent], mapping_cog2[p_brnn], mapping_cog2[p_svm]])

if TRAIN and TEST:
    ######### GET LABEL FOR EXERCISE QUESTIONS #########
    X_train1, Y_train1, X_test1, Y_test1 = get_data_for_cognitive_classifiers(threshold=[0.15, 0.25], what_type=['ada', 'bcl', 'os'], split=0.8, include_keywords=False, keep_dup=False, shuffle=False)

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

    ptest_maxent = []
    for x in [features(X1[i]) for i in range(len(X1))]:
        p_dict = clf_maxent.prob_classify(x)._prob_dict
        probs = np.array([p_dict[x] for x in range(6)])
        probs = np.exp(probs) / np.sum(np.exp(probs))
        ptest_maxent.append(probs)

    ptest_brnn = []
    for x in sent_to_glove(X1, w2v):
        probs = clf_brnn.predict_proba(clip(x))
        probs = [x[0] for x in probs]
        probs = np.exp(probs) / np.sum(np.exp(probs))
        ptest_brnn.append(probs)

    ptest_brnn = np.array(ptest_brnn)

    print('Loaded data for voting system')

    X = np.hstack((ptest_svm, ptest_maxent, ptest_brnn)) # concatenating the vectors
    Y = np.array(Y1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    ###### NEURAL NETWORK BASED VOTING SYSTEM ########
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(32, 16), batch_size=16, learning_rate='adaptive', learning_rate_init=0.001, verbose=True)
    clf.fit(x_train, y_train)
    print('ANN training completed')
    y_real, y_pred = y_test, clf.predict(x_test)

    joblib.dump(clf, 'models/cog_ann_voter.pkl')

    print('Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred) * 100))

    y_pred_svm = []
    y_pred_maxent = []
    y_pred_brnn = []

    for x in x_test:
        y_pred_svm.append(np.argmax(x[:6]))
        y_pred_maxent.append(np.argmax(x[6:12]))
        y_pred_brnn.append(np.argmax(x[12:]))

    print('SVM-GloVe Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred_svm) * 100))
    print('MaxEnt Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred_maxent) * 100))
    print('BiRNN Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred_brnn) * 100))

'''
######### PCA TO GET AGGREGATE OUTPUTS (deprecate in favour of AdaBoost) #########
X_data = []
pred_maxent = []
pred_svm = []
pred_brnn = []
with open('datasets/SO_Questions_Cog_Prediction.csv', 'r', encoding="utf-8") as csvfile:
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        if i == 0:
            continue
        question, p_maxent, p_brnn, p_svm = row
        X_data.append(question)
        pred_maxent.append(mapping_cog[p_maxent])
        pred_brnn.append(mapping_cog[p_brnn])
        pred_svm.append(mapping_cog[p_svm])

data = np.hstack((np.array(pred_maxent).reshape(-1, 1),
                  np.array(pred_brnn).reshape(-1, 1),
                  np.array(pred_svm).reshape(-1, 1)))

pca = PCA(n_components=3)
pca.fit_transform(data)
v = pca.explained_variance_ratio_
pred_agg = np.array(list(map(round, np.sum(data * v, axis=1))))

with open('datasets/SO_Questions_Cog_Prediction.csv', 'w', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Question', 'Cog(MaxEnt)', 'Cog(BiRNN)', 'Cog(SVM-GloVe)', 'Cog(Aggregate)'])
    for q, p_maxent, p_brnn, p_svm, p_agg in zip(X_data, pred_maxent, pred_brnn, pred_svm, pred_agg):
        csvwriter.writerow([q, mapping_cog2[p_maxent], mapping_cog2[p_brnn], mapping_cog2[p_svm], mapping_cog2[p_agg]])


######### TEST ACCURACY OF PCA METHOD #########


data = np.hstack((np.array(ptest_maxent).reshape(-1, 1),
                  np.array(ptest_brnn).reshape(-1, 1),
                  np.array(ptest_svm).reshape(-1, 1)))

print('Predictions acquired')

pca = PCA(n_components=3)
pca.fit_transform(data)
v = pca.explained_variance_ratio_
pred_agg = np.array(list(map(round, np.sum(data * v, axis=1))))

print('PCA fitting completed')

correct = 0
for i, p in enumerate(pred_agg):
    if p == Y_test1[i]:
        correct += 1

print('Aggregate accuracy: {:.2f}%'.format(correct / len(Y_test1) * 100.0))
'''

