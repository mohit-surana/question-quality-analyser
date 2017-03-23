'''
python3 struct_svm_ada.py

svm_multiclass/svm_multiclass_learn -c 5000 datasets/train_ada_cog.dat models/model_ada_cog.dat

svm_multiclass/svm_multiclass_classify datasets/test_ada_cog.dat models/model_ada_cog.dat datasets/predictions_ada_cog.dat

svm_multiclass/svm_multiclass_learn -c 5000 datasets/train_ada_know.dat models/model_ada_know.dat

svm_multiclass/svm_multiclass_classify datasets/test_ada_know.dat models/model_ada_know.dat datasets/predictions_ada_know.dat

If you get an error 13 - permission denied, run make inside svm_multiclass

'''

import codecs
import csv
import numpy as np
import pickle
import random
import re
import subprocess

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer

from utils import clean

PREPARE_VOCAB = False
TRAIN_CLASSIFIER = False
FILTERED = True

filtered_suffix = '_filtered' if FILTERED else ''

def prepare_file(filename, X, Y):
    with open(filename, 'w') as f:
        for i in range(len(X)):
            f.write(str(Y[i] + 1) + ' '),
            for j in range(len(X[i])):
                if(X[i][j] != 0.0):
                    f.write('%d:%f ' % (j+1, X[i][j]))
            f.write('\n')

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}

X = []
Y_cog = []
Y_know = []

# Uncomment for python2 usage
# reload(sys)
# sys.setdefaultencoding('utf8')

if(PREPARE_VOCAB or TRAIN_CLASSIFIER):
    freq = dict()
    with codecs.open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile.read().splitlines()[1:])
        for row in csvreader:
            sentence, label_cog, label_know = row
            m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
            sentence = m.groups()[2]
            label_cog = label_cog.split('/')[0]
            label_know = label_know.split('/')[0]
            clean_sentence = clean(sentence)
            for word in clean_sentence:
                freq[word] = freq.get(word, 0) + 1
            X.append(clean_sentence)
            Y_cog.append(mapping_cog[label_cog])
            Y_know.append(mapping_know[label_know])

    with codecs.open('datasets/BCLs_Question_Dataset.csv', 'r', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile.read().splitlines())
        for row in csvreader:
            sentence, label_cog = row
            clean_sentence = clean(sentence)
            if(PREPARE_VOCAB):
                for word in clean_sentence:
                    freq[word] = freq.get(word, 0) + 1
            X.append(clean_sentence)
            Y_cog.append(mapping_cog[label_cog])
            # TODO: Label
            Y_know.append(1)

    vocab = {word for word in freq if freq[word]> (1 if FILTERED else 0)}
    vocab_list = list(vocab)
    if(PREPARE_VOCAB):
        pickle.dump(vocab, open("models/vocab%s.pkl" % (filtered_suffix, ), 'wb'))
        pickle.dump(vocab_list, open("models/vocab_list%s.pkl" % (filtered_suffix, ), 'wb'))

vocab = pickle.load(open("models/vocab%s.pkl" % (filtered_suffix, ), 'rb'))
vocab_list = pickle.load(open("models/vocab_list%s.pkl" % (filtered_suffix, ), 'rb'))
vocab_size = len(vocab_list)

def train(X, Y, model_name='ada_cog'):
    dataset = list(zip(X,Y))
    random.shuffle(dataset)
    X, Y = zip(*dataset)

    X = np.array(X)
    Y = np.array(Y)

    X_vec = []
    for i in range(len(X)):
        sentence = X[i]
        X_vec.append(np.zeros((vocab_size, ), dtype=np.int32))
        for j in range(len(sentence)):
            word = sentence[j]
            if(word in vocab):
                X_vec[i][vocab_list.index(word)] += 1

    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(X_vec)

    X = tfidf.toarray()

    prepare_file('datasets/train_%s%s.dat' % (model_name, filtered_suffix), X[:(7*len(X))//10], Y[:(7*len(X))//10])
    prepare_file('datasets/test_%s%s.dat' % (model_name, filtered_suffix), X[(7*len(X))//10:], Y[(7*len(X))//10:])

    subprocess.call(['svm_multiclass/svm_multiclass_learn', '-c', '5000', 'datasets/train_%s%s.dat' % (model_name, filtered_suffix, ), 'models/model_%s%s.dat' % (model_name, filtered_suffix, )])

    subprocess.call(['svm_multiclass/svm_multiclass_classify', 'datasets/test_%s%s.dat' % (model_name, filtered_suffix, ), 'models/model_%s%s.dat' % (model_name, filtered_suffix, ), 'datasets/predictions_%s.dat' % (model_name, )])

if(TRAIN_CLASSIFIER):
    train(X, Y_cog)

def get_cognitive_probs(question):
    clean_question = clean(question)

    vec = np.zeros((vocab_size, ), dtype=np.int32)
    for j in range(len(clean_question)):
        word = clean_question[j]
        if(word in vocab_list):
            vec[vocab_list.index(word)] += 1

    transformer = joblib.load('models/tfidf_transformer%s.pkl' % (filtered_suffix, ))
    tfidf = transformer.fit_transform([vec])
    X = tfidf.toarray()

    prepare_file('datasets/test_ada_cog_sample.dat', X, [0])

    subprocess.call(['svm_multiclass/svm_multiclass_classify', 'datasets/test_ada_cog_sample.dat', 'models/model_ada_cog%s.dat' % (filtered_suffix, ), 'datasets/predictions_ada_cog_sample.dat'])

    with open('datasets/predictions_ada_cog_sample.dat', 'r') as f:
        line = f.read().split('\n')[0]
        label, probs = int(line.split()[0]) - 1, line.split()[1:]
        probs = [float(x) for x in probs]
        probs = np.array(probs)

        probs = abs(1 / probs)
        probs = np.exp(probs) / np.sum(np.exp(probs))

        for i in range(label + 1, 6):
            probs[i] = 0.0

        return probs

'''
with open('datasets/train_ada_know.dat', 'w') as f:
    for i in range((7*len(X))//10):
        f.write(str(Y_know[i] + 1) + ' '),
        for j in range(len(X[i])):
            if(X[i][j] != 0.0):
                f.write('%d:%f ' % (j+1, X[i][j]))
        f.write('\n')

with open('datasets/test_ada_know.dat', 'w') as f:
    for i in range((7*len(X))//10, len(X)):
        f.write(str(Y_know[i] + 1) + ' '),
        for j in range(len(X[i])):
            if(X[i][j] != 0.0):
                f.write('%d:%f ' % (j+1, X[i][j]))
        f.write('\n')
'''
