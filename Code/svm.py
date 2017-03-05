import csv
import random
import re
import numpy as np
from nltk import word_tokenize
from sklearn import svm
from sklearn.externals import joblib

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation))
#see documentation here: http://docs.python.org/2/library/string.html

from nltk.corpus import stopwords
#if not word in stopwords.words('english'): # Loss of crucial words

def clean(sentence):
    sentence = sentence.lower()

    final_sentence = []
    for word in word_tokenize(sentence):
        word = regex.sub(u'', word)
        if not (word == u'' or word == ''):
            word = wordnet.lemmatize(word)
            word = porter.stem(word)
            #word = snowball.stem(word)
            final_sentence.append(word)
    return final_sentence

def print_sentence(sentence):
    for word in sentence:
        if(word == vocab_size):
            break
        print(vocab_list[word] ,)
    print()

mapping = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}

vocab = set()

X = []
Y = []
max_len = 0
with open('datasets/BCLs_Question_Dataset.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        sentence, label = row
        clean_sentence = clean(sentence)
        for word in clean_sentence:
            vocab.add(word)
        X.append(clean_sentence)
        Y.append(mapping[label])

vocab_list = list(vocab)
vocab_size = len(vocab_list)

def get_cognitive_probs(question):
    clean_question = clean(question)

    vec = np.zeros((vocab_size, ), dtype=np.int32)
    for j in range(len(clean_question)):
        word = clean_question[j]
        if(word in vocab_list):
            vec[vocab_list.index(word)] += 1

    clf = joblib.load('models/svm_model.pkl')
    probs = clf.decision_function([vec])
    probs = abs(1/probs[0])

    label = clf.predict([vec])[0]

    probs = np.exp(probs)/np.sum(np.exp(probs))

    for i in range(label + 1, 6):
        probs[i] = 0.0

    return probs


if __name__ == '__main__':
    # TODO: Requires massive cleanup
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
            X_vec[i][vocab_list.index(word)] += 1


    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(X_vec)

    #from sklearn.feature_extraction.text import TfidfVectorizer
    #vectorizer = TfidfVectorizer(min_df=1)
    #print(vectorizer.fit_transform(X_vec[0]).toarray())

    X = tfidf.toarray()

    # 72% accuracy
    # Default 1k for max_iter
    # can adjust tol, C -> http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

    '''
    clf = svm.LinearSVC(max_iter=1000)
    clf.fit(X[:(7*len(X))//10], Y[:(7*len(X))//10])
    predictions = clf.predict(X[(7*len(X))//10:])
    targets = Y[(7*len(X))//10:]
    '''

    clf = svm.LinearSVC()
    clf.fit(X[:(7*len(X))//10], Y[:(7*len(X))//10])

    joblib.dump(clf, 'svm_model.pkl')
    clf = joblib.load('svm_model.pkl')

    predictions = clf.decision_function(X[(7*len(X))//10:])
    # This will return noOfSamples x noOfClasses probs

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
