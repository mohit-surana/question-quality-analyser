import numpy as np
import codecs
import random
import csv
from utils import clean
import re
import dill
import pickle
from collections import defaultdict
from sklearn.externals import joblib
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib


mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}
X = []
Y_cog = []
Y_know = []
TRAIN_SVM_GLOVE = True


domain = pickle.load(open('resources/domain_2.pkl',  'rb'))

keywords = set()
for k in domain:
    keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))

def lamb1(x):
    return x

def lamb2():
    return gVar

global gVar
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec['gay'])
        
    def fit(self, X, y):
        global gVar
        tfidf = TfidfVectorizer(analyzer=lamb1)
        tfidf.fit(X)

        max_idf = max(tfidf.idf_)
        gVar = max_idf
        self.word2weight = defaultdict(lamb2,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec and w in keywords] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
               
################ BEGIN LOADING DATA ################

with open("models/glove.6B.50d.txt", "r") as lines:
    w2v = {line.split()[0]: np.array(list(map(float, line.split()[1:])))
           for line in lines}

with open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r') as csvfile:
    all_rows = csvfile.read().splitlines()[1:]
    csvreader = csv.reader(all_rows)
    for row in csvreader:
        sentence, label_cog, label_know = row
        m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
        sentence = m.groups()[2]
        label_cog = label_cog.split('/')[0]
        label_know = label_know.split('/')[0]
        clean_sentence = clean(sentence.encode('utf-8').decode('utf-8'), stem=False)
        X.append(clean_sentence)
        Y_cog.append(mapping_cog[label_cog])
        Y_know.append(mapping_know[label_know])

with open('datasets/BCLs_Question_Dataset.csv', 'r') as csvfile:
    all_rows = csvfile.read().splitlines()[1:]
    csvreader = csv.reader(all_rows) 
    for row in csvreader:
        sentence, label_cog = row
        clean_sentence = clean(sentence.encode('utf-8').decode('utf-8'), stem=False)
        X.append(clean_sentence)
        Y_cog.append(mapping_cog[label_cog])
        # TODO: Label
        Y_know.append(1)

dataset = list(zip(X, Y_cog))
random.shuffle(dataset)
X_train = []
Y_train = []
X_test = []
Y_test = []

for x, y in dataset[:len(dataset) * 7//10]:
    X_train.append(x)
    Y_train.append(y)

for x, y in dataset[len(dataset) * 7//10:]:
    X_test.append(x)
    Y_test.append(y)

domain_keywords = pickle.load(open('resources/domain.pkl', 'rb'))
for key in domain_keywords:
    for word in domain_keywords[key]:            
        X_train.append([word])
        Y_train.append(mapping_cog[key])

print('Data Loaded and Processed')

################ BEGIN TRAINING CODE ################

if TRAIN_SVM_GLOVE:
    print('Fitting Started')
    classify = Pipeline([ ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
                          ('svc', svm.SVC(kernel='linear')) ])

    classify.fit(X_train, Y_train)
    print('Fitting Done')

    joblib.dump(classify, 'models/glove_svm_model.pkl') 
    print('Training done')

################ BEGIN TESTING CODE ################

classify = joblib.load('models/glove_svm_model.pkl')

print('Testing Started')
print('Accuracy: {:.3f}%'.format(classify.score(X_test, Y_test) * 100))

