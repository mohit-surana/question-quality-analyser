'''
1D prediction for knowledge
hyperparam
toggle para/section
'''

import math
import nltk
import numpy as np
import os
import pprint
import re
import pickle
import random
import csv
import pprint
import platform

from nltk import stem
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from pandas import DataFrame

import sys

knowledge_mapping = {'Metacognitive': 3, 'Procedural': 2, 'Conceptual': 1, 'Factual': 0}
knowledge_mapping2 = {v : k for k, v in knowledge_mapping.items()}

def __get_knowledge_level(question, subject='ADA'):
    classifier = pickle.load(open('models/Nsquared/%s/nsquared.pkl' % (subject, ), 'rb'))
    all_classes = sorted(list(set(list(classifier.data['class'].values))))

    with open('resources/%s/__Sections.csv' % (subject, ), 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows_read = 0

        X = {}
        for row in csvreader:
            if rows_read == 0:
                rows_read += 1
                continue
            if subject == 'ADA':
                s_no, _, section, _ = row
            else:
                id, s_no, level, section, pg_no = row
            X[section] = (section, rows_read - 1, 0)
            rows_read += 1

    section_wise_question_probs = []
    for i, prob in enumerate(classifier.classify(question)):
        #print('Question:{}'.format(question[i]))
        for c, p in __probs_to_classes_dict(prob, all_classes).items():
            X[c] = (X[c][0], X[c][1], p)

        section_wise_question_probs = list(filter(lambda x: x[-1] != 0, sorted(X.values(), key=lambda x: x[2], reverse=True)))

    highest_prob = section_wise_question_probs[0][-1]

    hardcoded_matrix = [1.0, 0.6, 0.3, 0.1, 0]

    for i, r in enumerate(zip(hardcoded_matrix[1:], hardcoded_matrix[:-1])):
        if r[0] <= highest_prob < r[1]:
            return i, highest_prob


def get_knowledge_probs(question, subject):
    level, highest_prob = __get_knowledge_level(question, subject)
    #print(level, highest_prob)
    probs = [0.0] * 4
    for i in range(level):
        # probs[i] = (i + 1) * highest_prob / (level * (level + 1) / 2)
        probs[i] = (i + 1) * highest_prob / (level + 1)
    probs[level] = highest_prob
    return probs

def __probs_to_classes(probs, all_classes):
    probs = enumerate(probs)
    probs = sorted(probs, key=lambda x: x[1], reverse=True)
    #classes = {}
    classes = []
    for i, p in probs:
        #classes[all_classes[i]] = p
        classes.append((all_classes[i], p))

    return classes

def __probs_to_classes_dict(probs, all_classes):
    probs = enumerate(probs)
    probs = sorted(probs, key=lambda x: x[1], reverse=True)
    classes = {}
    for i, p in probs:
        classes[all_classes[i]] = p

    return classes


class DocumentClassifier:
    stemmer = stem.porter.PorterStemmer()
    wordnet = WordNetLemmatizer()

    punkt = {',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '.', '?', '!', '`', '|', '-', '=', '+', '_', '>', '<'}

    if(platform.system() == 'Windows'):
        stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r', encoding='utf8').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
    else:
        stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
    stopwords.update(punkt)

    def __preprocess(self, text):
        text = re.sub('-\n', '', text).lower()
        text = re.sub('\n', ' ', text)
        text = re.sub('[^a-z ]', '', text)
        tokens = [word for word in nltk.word_tokenize(text)]  
        return ' '.join(list(set([self.wordnet.lemmatize(i) for i in tokens if i not in self.stopwords])))

    def __init__(self, subject, skip_files=[]):
        self.subject = subject
        self.pipeline = Pipeline([
                        ('vectorizer', TfidfVectorizer(sublinear_tf=True,
                                                       max_df=0.5,
                                                       ngram_range=(1, 2),
                                                       stop_words='english',
                                                       strip_accents='unicode', 
                                                       norm='l2',
                                                       decode_error="ignore")),
                        ('classifier', MultinomialNB(alpha=.01)) ])

        X = [] #DataFrame({'text': [], 'class': []})
        Y = []

        count = 0
        for file_name in sorted(os.listdir('resources/%s' % (self.subject, ))):
            with open(os.path.join('resources', self.subject, file_name), encoding='latin-1') as f:
                content = re.split('\n[\s]*Exercise', f.read())[0]
                title = content.split('\n')[0]
                if len([1 for k in skip_files if (k in title or k in file_name)]):
                    continue

                '''
                sentences = nltk.sent_tokenize(content)

                
                for sentence in sentences:
                    text = self.__preprocess(sentence)
                    if text and len(text.split()) > 2:
                '''

                X.append(self.__preprocess(content))
                Y.append(content.split('\n')[0])


                print('Loaded and processed', file_name)
        
        X_data = list(zip(X, Y))
        random.shuffle(X_data)
        X = [x[0] for x in X_data]
        Y = [x[1] for x in X_data]

        self.classes = set(Y) 

        print('\nFitting data into the classifier')
        self.pipeline.fit(X, Y)

    def classify(self, data):
        if type(data) != type([]):
            data = [data]

        return self.pipeline.predict([self.__preprocess(d) for d in data])

    def save(self, file_name):
        pickle.dump(self, open(file_name, 'wb'))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Not enough arguments')

    subject = sys.argv[1]

    TRAIN = True
    if TRAIN:
        classifier = DocumentClassifier(subject=subject, skip_files={'__', '.DS_Store', 'Key Terms, Review Questions, and Problems', 'Recommended Reading and Web Sites', 'Recommended Reading', 'Summary', 'Exercises', 'Introduction'})
        classifier.save('models/Nsquared/%s/nsquared.pkl' % (subject, ))

    classifier = pickle.load(open('models/Nsquared/%s/nsquared.pkl' % (subject, ), 'rb'))

    all_classes = sorted(list(set(list(classifier.classes))))

    questions = ["""Having some problems implementing a quicksort sorting algorithm in java.""",
    """Give an example that shows that the approximation sequence of Newton's method may diverge. """,
    """Find the number of comparisons made by the sentinel version of linear search  b in the worst case. in the average case if the probability of a successful search is p (0 p 1).""",
    """ Write a brute force pattern matching program for playing the game Battleship on the computer."""]

    for i, question in enumerate(questions):
        print('Question', i + 1)
        for sentence in nltk.sent_tokenize(question):
            prob = classifier.classify(sentence)
            print('[{}]'.format(prob))
        print()
    '''
    for i, prob in enumerate(classifier.classify(questions )):
        print('Question', i + 1)
        for c, p in __probs_to_classes(prob, all_classes)[:2]:
            print(c, '[{}]'.format(p))
    '''
