import codecs
import csv
import numpy as np
import pickle
import random
import re
import string
import sys

from nltk import word_tokenize
from nltk.corpus import stopwords
#if not word in stopwords.words('english'): # Loss of crucial words
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

regex = re.compile('[%s]' % re.escape(string.punctuation))
#see documentation here: http://docs.python.org/2/library/string.html

PREPARE_VOCAB = False
TRAIN_CLASSIFIER = False

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

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}


X = []
Y_cog = []
Y_know = []

# Uncomment for python2 usage
# reload(sys)
# sys.setdefaultencoding('utf8')

if(PREPARE_VOCAB or TRAIN_CLASSIFIER):
    vocab = set()
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
                vocab.add(word)
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
                    vocab.add(word)
            X.append(clean_sentence)
            Y_cog.append(mapping_cog[label_cog])
            # TODO: Label
            Y_know.append(1)

    vocab_list = list(vocab)
    if(PREPARE_VOCAB):
        pickle.dump(vocab, open("models/vocab.pkl", 'wb'))
        pickle.dump(vocab_list, open("models/vocab_list.pkl", 'wb'))

vocab = pickle.load(open("models/vocab.pkl", 'rb'))
vocab_list = pickle.load(open("models/vocab_list.pkl", 'rb'))
vocab_size = len(vocab_list)

if(TRAIN_CLASSIFIER):
    dataset = list(zip(X,Y_cog))
    random.shuffle(dataset)
    X, Y_cog = zip(*dataset)

    X = np.array(X)
    Y_cog = np.array(Y_cog)

    X_vec = []
    for i in range(len(X)):
        sentence = X[i]
        X_vec.append(np.zeros((vocab_size, ), dtype=np.int32))
        for j in range(len(sentence)):
            word = sentence[j]
            X_vec[i][vocab_list.index(word)] += 1

    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(X_vec)

    X = tfidf.toarray()

    clf = svm.LinearSVC()
    clf.fit(X[:(7*len(X))//10], Y_cog[:(7*len(X))//10])

    joblib.dump(transformer, 'models/tfidf_transformer.pkl')
    joblib.dump(clf, 'models/svm_model.pkl')

    predictions = clf.decision_function(X[(7*len(X))//10:])

    predictions = [np.argmax(prediction) for prediction in predictions]
    targets = Y_cog[(7*len(X))//10:]

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


def get_cognitive_probs(question):
    clean_question = clean(question)

    vec = np.zeros((vocab_size, ), dtype=np.int32)
    for j in range(len(clean_question)):
        word = clean_question[j]
        if(word in vocab_list):
            vec[vocab_list.index(word)] += 1

    transformer = joblib.load('models/tfidf_transformer.pkl')
    tfidf = transformer.fit_transform([vec])
    X = tfidf.toarray()

    clf = joblib.load('models/svm_model.pkl')
    probs = clf.decision_function(X)
    probs = abs(1/probs[0])

    label = clf.predict([vec])[0]

    probs = np.exp(probs)/np.sum(np.exp(probs))

    for i in range(label + 1, 6):
        probs[i] = 0.0

    return probs

if __name__ == '__main__':
    pass
