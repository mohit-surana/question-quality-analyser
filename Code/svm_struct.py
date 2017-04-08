'''
python3 struct_svm_ada.py

svm_multiclass/svm_multiclass_learn -c 5000 datasets/StructSVM/train_ada_cog.dat models/SVM/Struct/model_ada_cog.dat

svm_multiclass/svm_multiclass_classify datasets/StructSVM/test_ada_cog.dat models/SVM/Struct/model_ada_cog.dat datasets/StructSVM/predictions_ada_cog.dat

svm_multiclass/svm_multiclass_learn -c 5000 datasets/StructSVM/train_ada_know.dat models/SVM/Struct/model_ada_know.dat

svm_multiclass/svm_multiclass_classify datasets/StructSVM/test_ada_know.dat models/SVM/Struct/model_ada_know.dat datasets/StructSVM/predictions_ada_know.dat

If you get an error 13 - permission denied, run make inside svm_multiclass

'''

import codecs
import csv
import dill
import numpy as np
import os
import pickle
import random
import re
import subprocess

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import clean
from utils import get_data_for_cognitive_classifiers

TRAIN_CLASSIFIER = True
FILTERED = True

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

def sent_to_glove(questions, w2v):
	questions_w2glove = []
	
	for question in questions:
		vec = []
		for word in question:
			if word in w2v:
				vec.append(w2v[word])
			else:
				vec.append(np.zeros(len(w2v['the'])))
		questions_w2glove.append(np.array(vec))
	
	return np.array(questions_w2glove)

def transform_to_glove(X, INPUT_SIZE=300):
    filename = 'glove.840B.%dd.txt' % INPUT_SIZE

    if not os.path.exists('resources/GloVe/%s_saved.pkl' % filename.split('.txt')[0]):
        print()
        with open('resources/GloVe/' + filename, "r", encoding='utf-8') as lines:
            w2v = {}
            for row, line in enumerate(lines):
                try:
                    w = line.split()[0]
                    if w not in vocabulary:
                        continue
                    vec = np.array(list(map(float, line.split()[1:])))
                    w2v[w] = vec
                except:
                    continue
                finally:
                    print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} GloVe vectors'.format(row + 1))
        
        dill.dump(w2v, open('resources/GloVe/%s_saved.pkl' % filename.split('.txt')[0], 'wb'))
    else:
        w2v = dill.load(open('resources/GloVe/%s_saved.pkl' % filename.split('.txt')[0], 'rb'))

    X_data = sent_to_glove(X, w2v)
    return X_data


def train(X, Y, model_name='ada_cog'):
    dataset = list(zip(X,Y))
    random.shuffle(dataset)
    X, Y = zip(*dataset)

    X = np.array(X)
    Y = np.array(Y)

    '''X_vec = transform_to_glove(X)
    print(X.shape)
    print(len(X[0])) # List
    print(X_vec.shape)
    print(X_vec[0].shape)
    print(X_vec[0][0].shape)
    print(X_vec[0].reshape(-1, 3).shape)'''
    
    # for i in range(len(X_vec)):
    #     X_vec[i] = np.average(X_vec[i].reshape(-1, 3))
    
    # transformer = TfidfTransformer(smooth_idf=True)
    # tfidf = transformer.fit_transform(X_vec)
     #joblib.save(transformer, 'models/SVM/Struct/tfidf_transformer.pkl')
    
    # X = tfidf.toarray()
    
    for i in range(len(X)):
        X[i] = ' '.join(X[i])
    
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    X = X_vec.toarray()
    
    prepare_file('datasets/StructSVM/train_%s.dat' % (model_name, ), X[:(7*len(X))//10], Y[:(7*len(X))//10])
    prepare_file('datasets/StructSVM/test_%s.dat' % (model_name, ), X[(7*len(X))//10:], Y[(7*len(X))//10:])

    subprocess.call(['svm_multiclass/svm_multiclass_learn', '-c', '5000', 'datasets/StructSVM/train_%s.dat' % (model_name, ), 'models/SVM/Struct/model_%s.dat' % (model_name, )])

    subprocess.call(['svm_multiclass/svm_multiclass_classify', 'datasets/StructSVM/test_%s.dat' % (model_name, ), 'models/SVM/Struct/model_%s.dat' % (model_name, ), 'datasets/StructSVM/predictions_%s.dat' % (model_name, )])

if(TRAIN_CLASSIFIER):
    X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers()
    train(X_train + X_test, Y_train + Y_test)

def get_cognitive_probs(question):
    clean_question = clean(question)

    vec = np.zeros((vocab_size, ), dtype=np.int32)
    for j in range(len(clean_question)):
        word = clean_question[j]
        if(word in vocab_list):
            vec[vocab_list.index(word)] += 1

    transformer = joblib.load('models/SVM/Struct/tfidf_transformer.pkl')
    tfidf = transformer.fit_transform([vec])
    
    X = tfidf.toarray()

    prepare_file('datasets/StructSVM/test_ada_cog_sample.dat', X, [0])

    subprocess.call(['svm_multiclass/svm_multiclass_classify', 'datasets/StructSVM/test_ada_cog_sample.dat', 'models/SVM/Struct/model_ada_cog.dat', 'datasets/StructSVM/predictions_ada_cog_sample.dat'])

    with open('datasets/StructSVM/predictions_ada_cog_sample.dat', 'r') as f:
        line = f.read().split('\n')[0]
        label, probs = int(line.split()[0]) - 1, line.split()[1:]
        probs = [float(x) for x in probs]
        probs = np.array(probs)

        probs = abs(1 / probs)
        probs = np.exp(probs) / np.sum(np.exp(probs))

        for i in range(label + 1, 6):
            probs[i] = 0.0

        return probs
