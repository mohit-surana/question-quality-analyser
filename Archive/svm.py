import codecs
import csv
import pickle
import random

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer

from utils import clean

PREPARE_VOCAB = True
TRAIN_CLASSIFIER = True
FILTERED = True

filtered_suffix = '_filtered' if FILTERED else ''

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
    '''
    with codecs.open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
        all_rows = csvfile.read().splitlines()[1:]
        csvreader = csv.reader(all_rows[:len(all_rows)*7//10])
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
    '''
    with codecs.open('datasets/BCLs_Question_Dataset.csv', 'r', encoding="utf-8") as csvfile:
        all_rows = csvfile.read().splitlines()[1:]
        csvreader = csv.reader(all_rows)  #csvreader = csv.reader(all_rows[:len(all_rows)*7//10])
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
            
    domain_keywords = pickle.load(open('resources/domain.pkl', 'rb'))
    for key in domain_keywords:
        for word in domain_keywords[key]:
            freq[word] = freq.get(word, 0) + 1
            X.append([word])
            Y_cog.append(mapping_cog[key])

    vocab = {word for word in freq if freq[word]> (1 if FILTERED else 0)}

    vocab_list = list(vocab)
    if(PREPARE_VOCAB):
        pickle.dump(vocab, open("models/vocab%s.pkl" % (filtered_suffix, ), 'wb'))
        pickle.dump(vocab_list, open("models/vocab_list%s.pkl" % (filtered_suffix, ), 'wb'))

vocab = pickle.load(open("models/vocab%s.pkl" % (filtered_suffix, ), 'rb'))
vocab_list = pickle.load(open("models/vocab_list%s.pkl" % (filtered_suffix, ), 'rb'))
vocab_size = len(vocab_list)

def train(X, Y, model_name='svm_model'):
    # NOTE: Prerequisites = vocab is ready
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
    print(X)
    clf = svm.SVC(kernel='rbf') #clf = svm.LinearSVC()
    clf.fit(X,Y)
    joblib.dump(transformer, 'models/tfidf_transformer%s.pkl' % (filtered_suffix, ))
    joblib.dump(clf, 'models/%s%s.pkl' % (model_name, filtered_suffix, ))

    '''
    # MOHIT, don't cry when you see this, your code is still here. :P
    clf.fit(X[:(7*len(X))//10], Y[:(7*len(X))//10])

    joblib.dump(transformer, 'models/tfidf_transformer%s.pkl' % (filtered_suffix, ))
    joblib.dump(clf, 'models/%s%s.pkl' % (model_name, filtered_suffix, ))

    predictions = clf.decision_function(X[(7*len(X))//10:])

    predictions = [np.argmax(prediction) for prediction in predictions]
    targets = Y[(7*len(X))//10:]

    correct = 0
    for i in range(len(predictions)):
        if(predictions[i] == targets[i]):
            correct += 1

    print('Overall accuracy:', correct/len(predictions))

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

    print('F1 Score:', f1_score(targets, predictions, average="macro"))
    # print(precision_score(targets, predictions, average="macro"))
    # print(recall_score(targets, predictions, average="macro"))
    print(classification_report(targets, predictions))
    # print(confusion_matrix(targets, predictions))
    '''
    
if(TRAIN_CLASSIFIER):
    train(X, Y_cog)

def get_cognitive_probs(question, model_name='svm_model'):
    clean_question = clean(question)

    vec = np.zeros((vocab_size, ), dtype=np.int32)
    for j in range(len(clean_question)):
        word = clean_question[j]
        if(word in vocab_list):
            vec[vocab_list.index(word)] += 1

    transformer = joblib.load('models/tfidf_transformer%s.pkl' % (filtered_suffix, ))
    tfidf = transformer.fit_transform([vec])
    X = tfidf.toarray()

    clf = joblib.load('models/%s%s.pkl' % (model_name, filtered_suffix, ))
    probs = clf.decision_function(X)
    probs = abs(1 / probs[0])

    label = clf.predict([vec])[0]
    probs = np.exp(probs) / np.sum(np.exp(probs))

    for i in range(label + 1, 6):
        probs[i] = 0.0

    return probs
    
def get_labels_batch(questions, model_name='svm_model'):
    
    labels = []
    probabs = []
    for question in questions:
        clean_question = clean(question)

        vec = np.zeros((vocab_size, ), dtype=np.int32)
        for j in range(len(clean_question)):
            word = clean_question[j]
            if(word in vocab_list):
                vec[vocab_list.index(word)] += 1

        transformer = joblib.load('models/tfidf_transformer%s.pkl' % (filtered_suffix, ))
        tfidf = transformer.fit_transform([vec])
        X = tfidf.toarray()

        clf = joblib.load('models/%s%s.pkl' % (model_name, filtered_suffix, ))
        probs = clf.decision_function(X)
        probs = abs(1 / probs[0])

        label = clf.predict([vec])[0]
        labels.append(label)
        probs = np.exp(probs) / np.sum(np.exp(probs))
    
        for i in range(label + 1, 6):
            probs[i] = 0.0
        
        probabs.append(probs)

    return probabs, labels
    
def get_prediction(labels, targets):
    count = 0
    result = list(zip(labels, targets))
    for res in result:
        if res[0] == res[1]:
            count += 1
    return count/len(result)
    
if __name__ == '__main__':
    pass
