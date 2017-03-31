import classifier as Nsq
import codecs
import csv
import docsim_lda
# import docsim_lsa
import numpy as np
import pickle
import re
# import struct_svm_ada
import svm

from classifier import DocumentClassifier
from scipy import linalg
from utils import clean

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}

X = []
Y_cog = []
Y_know = []

with codecs.open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
    csvreader = csv.reader(csvfile.read().splitlines()[1:])
    # NOTE: This is used to skip the first line containing the headers
    for row in csvreader:
        sentence, label_cog, label_know = row
        m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
        sentence = m.groups()[2]
        label_cog = label_cog.split('/')[0]
        label_know = label_know.split('/')[0]
        # sentence = clean(sentence)
        # NOTE: Commented the above line because the cleaning mechanism is different for knowledge and cognitive dimensions
        X.append(sentence)
        Y_cog.append(mapping_cog[label_cog])
        Y_know.append(mapping_know[label_know])

TRAIN = False
TEST = True
EPOCHS = 1000
# EPOCHS = 1
Ks, Cs = None, None

def prepare_y(Y_cog, Y_know):
    Y = []
    for i in range(len(Y_cog)):
        Y.append([])
        for j in range(len(Y_cog)):
            if(Y_cog[i] == Y_cog[j] and Y_know[i] == Y_know[j]):
                Y[i].append(1)
            else:
                Y[i].append(0)
        Y[i] = np.array(Y[i])
    return np.array(Y)

if(TRAIN or TEST):
    Ks = []
    Cs = []
    for question in X:
        K = Nsq.get_knowledge_probs(question, 'ADA')
        # sims, K = docsim_lda.get_vector('n', question, 'tfidf')
        # C = struct_svm_ada.get_cognitive_probs(question)
        C = svm.get_cognitive_probs(question)
        Ks.append(np.array(K))
        Cs.append(np.array(C))

    Ks_train = np.array(Ks[:7*len(Ks)//10])
    Cs_train = np.array(Cs[:7*len(Ks)//10])
    Ks_test = np.array(Ks[7*len(Ks)//10:])
    Cs_test = np.array(Cs[7*len(Ks)//10:])
    
    Y_train = prepare_y(Y_cog[:7*len(Ks)//10], Y_know[:7*len(Ks)//10])
    Y_test = prepare_y(Y_cog[7*len(Ks)//10:], Y_know[7*len(Ks)//10:])

if(TRAIN):
    inv = linalg.lstsq
    
    temp, _, _, _ =  inv(Ks_train, Y_train)
    # W, _, _, _ = inv(temp, Cs_train.T)
    W, _, _, _ = inv(temp.T, Cs_train)
    
    pickle.dump(W, open("models/ADA_W_adj.pkl", 'wb'))

W = pickle.load(open("models/ADA_W_adj.pkl", 'rb'))

def pred2label(x):
    val = round(x)
    if(val < 0):
        val = 0
    if(val > 23):
        val = 23
    return int(val)

targets = [y_cog + 6 * y_know for y_cog, y_know in zip(Y_cog, Y_know)]
predictions = []

correct = 0
total = 0

if(TEST):
    for K, C, y in list(zip(Ks, Cs, targets))[7*len(Ks)//10:]:
        prediction = np.dot(np.dot(K.reshape(1, 4), W), C.reshape(1, 6).T)
        prediction = pred2label(prediction[0][0])
        print(prediction, y)
        if(prediction == y):
            correct += 1
        total += 1

    print('Accuracy:', (correct / total) )
