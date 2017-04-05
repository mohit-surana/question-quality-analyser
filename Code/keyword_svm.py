import codecs
import csv
import numpy as np
import pickle
import random
import re
import sys

from sklearn import svm
from sklearn.externals import joblib
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from utils import clean

TRAIN_CLASSIFIER = True

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}


X = []
Y_cog = []
Y_know = []

# Uncomment for python2 usage
# reload(sys)
# sys.setdefaultencoding('utf8')
def train(X, Y, model_name='svm_model'):
    # NOTE: Prerequisites = vocab is ready
    dataset = list(zip(X,Y))
    random.shuffle(dataset)
    X, Y = zip(*dataset)

    Y = np.array(Y)

    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(X)
    print(Y)
    #print(X)
    clf = svm.SVC(kernel='rbf') #clf = svm.LinearSVC()
    clf.fit(X,Y)
    joblib.dump(vectorizer, 'models/%s_vectorizer.pkl' % (model_name))
    joblib.dump(clf, 'models/%s_rbf_vect.pkl' % (model_name))


if TRAIN_CLASSIFIER:
    domain_keywords = pickle.load(open('resources/domain.pkl', 'rb'))
    for key in domain_keywords:
        for word in domain_keywords[key]:
            X.append(word)
            Y_cog.append(mapping_cog[key])
    train(X, Y_cog)





def get_cognitive_probs(question, model_name='svm_model'):

    clean_question = clean(question, return_as_list = False)
        
    vectorizer = joblib.load('models/%s_vectorizer.pkl' % (model_name))
    
    X = vectorizer.transform([clean_question]).toarray()
    print(X)
    clf = joblib.load('models/%s_rbf_vect.pkl' % (model_name))
    probs = clf.decision_function(X)
    #probs = abs(1 / probs[0])
    #print(probs, abs(1 / probs[0]))
    label = clf.predict(X)
    
    #print(label)
    
    return label
    
def get_labels_batch(questions, model_name='svm_model'):
    clean_question = []
    for question in questions:
        clean_question.append(clean(question, return_as_list = False))
        
    vectorizer = joblib.load('models/%s_vectorizer.pkl' % (model_name))
    
    X = vectorizer.transform(clean_question).toarray()
    print(X)
    clf = joblib.load('models/%s_rbf_vect.pkl' % (model_name))
    probs = clf.decision_function(X)
    #probs = abs(1 / probs[0])
    #print(probs, abs(1 / probs[0]))
    label = clf.predict(X)
    
    #print(label)
   
    

    return probs, label
    
def get_prediction(labels, targets):
    count = 0
    result = list(zip(labels, targets))
    for res in result:
        if res[0] == res[1]:
            count += 1
    return count/len(result)
    
if __name__ == '__main__':
    X_test = []
    Y_test = []
    #print(get_cognitive_probs('What is horners rule?'))
    with codecs.open('datasets/OS_Exercise_Questions_Relabelled.csv', 'r', encoding="utf-8") as csvfile:
        all_rows = csvfile.read().splitlines()[1:]
        csvreader = csv.reader(all_rows)  #csvreader = csv.reader(all_rows[:len(all_rows)*7//10])
        for row in csvreader:
            sentence = row[0]
            label_cog = row[-1]
            X_test.append(sentence)
            Y_test.append(int(label_cog))
    prob, label = get_labels_batch(X_test)
    print(Y_test, label)
    count = 0
    for i in range(len(Y_test)):
        if(Y_test[i] == label[i]):
            count += 1
    print('Accuracy', count/len(Y_test))