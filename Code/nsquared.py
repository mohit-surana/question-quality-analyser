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
import csv
import pprint
import platform
import random
import copy

from nltk import stem

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier

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

def probs_to_classes(probs, all_classes):
    probs = enumerate(probs)
    probs = sorted(probs, key=lambda x: x[1], reverse=True)
    #classes = {}
    classes = []
    for i, p in probs:
        #classes[all_classes[i]] = p
        classes.append((all_classes[i], p))

    return classes

def probs_to_classes_dict(probs, all_classes):
    probs = enumerate(probs)
    probs = sorted(probs, key=lambda x: x[1], reverse=True)
    classes = {}
    for i, p in probs:
        classes[all_classes[i]] = p

    return classes

def get_questions(subject, skip_files, shuffle=True):
    exercise_content = {}
    for filename in sorted(os.listdir('./resources/%s' %subject)):
        with open('./resources/%s/'%subject + filename, encoding='latin-1') as f:
            contents = f.read()
            title = contents.split('\n')[0].strip()
            if len([1 for k in skip_files if (k in title or k in filename)]):
                continue

            match = re.search(r'\n[\s]*Exercises[\s]+([\d]+\.[\d]+)[\s]*(.*)', contents, flags=re.M | re.DOTALL) 

            if match:
                exercise_content[title] = '\n' + match.group(2).split('SUMMARY')[0]

        X_data, Y_data = [], []
        for e in exercise_content:
            for question in re.split('[\n][\s]*[\d]+\.', exercise_content[e].strip(), flags=re.M | re.DOTALL):
                if len(question) > 0:
                    X_data.append(re.sub('\n', ' ', re.sub('1\.', '', question.strip()), flags=re.M | re.DOTALL))
                    Y_data.append(e)

    if shuffle:
        X = list(zip(X_data, Y_data))
        random.shuffle(X)
        X_data = [x[0] for x in X]
        Y_data = [x[1] for x in X]

    return X_data, Y_data

class DocumentClassifier:
    stemmer = stem.porter.PorterStemmer()
    wordnet = WordNetLemmatizer()

    punkt = {',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '.', '?', '!', '`', '|', '-', '=', '+', '_', '>', '<'}

    if(platform.system() == 'Windows'):
        stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r', encoding='utf8').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
    else:
        stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
    stopwords.update(punkt)

    def __preprocess(self, text, remove_stopwords=True):
        text = re.sub('-\n', '', text).lower()
        text = re.sub('\n', ' ', text)
        text = re.sub('[^a-z ]', '', text)
        tokens = [word for word in text.split()]  
        if remove_stopwords:
            return ' '.join([self.wordnet.lemmatize(i) for i in tokens if i not in self.stopwords])
        else:
            return ' '.join([self.wordnet.lemmatize(i) for i in tokens])

    def __init__(self, subject, skip_files=[]):
        self.subject = subject

        self.chapter_map = []
        with open('resources/%s/__Sections.csv' %subject) as f:
            rows = f.read().splitlines()
            csvreader = csv.reader(rows)
            for row in csvreader:
                if row[1] in ['1', '2']:
                    self.chapter_map.append({'chapter' : int(row[0].split('.')[0]), 'level' : int(row[1]), 'section' : row[2]})

        self.section_map = {x['section'] : x['chapter'] for x in self.chapter_map if x['level'] == 2} # convert to a lookup of sections
        self.chapter_map = {x['chapter'] : x['section'] for x in self.chapter_map if x['level'] == 1} # convert to a lookup of chapters

        self.data = DataFrame({'text': [], 'class': []})
        self.section_data = { k : DataFrame({'text': [], 'class': []}) for k in self.chapter_map.values() }

        for file_name in sorted(os.listdir('resources/%s' % (self.subject, ))):
            with open(os.path.join('resources', self.subject, file_name), encoding='latin-1') as f:
                content = f.read() #re.split('\n[\s]*Exercise', f.read())[0] 
                title = content.split('\n')[0]
                if len([1 for k in skip_files if (k in title or k in file_name)]):
                    continue

                sentences = nltk.sent_tokenize(content)

                rows_section, rows_chapter = [], []
                for sentence in sentences:
                    text = self.__preprocess(sentence, remove_stopwords=True)
                    text2 = self.__preprocess(sentence, remove_stopwords=True)
                    if text and len(text.split()) > 2:
                        if title in self.chapter_map.values():
                            chapter = title
                        else:
                            chapter = self.chapter_map[self.section_map[content.split('\n')[0]]]
                            rows_section.append({'text' : text2, 'class' : content.split('\n')[0] })
                        
                        rows_chapter.append({'text' : text, 'class' : chapter })

                self.data = self.data.append(DataFrame(rows_chapter))
                self.section_data[chapter] = self.section_data[chapter].append(DataFrame(rows_section))

                print('Loaded and processed', file_name)

        X_questions, Y_questions = get_questions(subject, skip_files, shuffle=False)

        '''
        rows = []
        for x, y in zip(X_questions, Y_questions):
            chapter = self.chapter_map[self.section_map[y]]
            for sentence in nltk.sent_tokenize(x):
                self.section_data[chapter] = self.section_data[chapter].append(DataFrame([{'text' : self.__preprocess(sentence, remove_stopwords=True), 'class' : y}]))
        '''

        self.data = self.data.sample(frac=1).reset_index(drop=True)
        for k in self.section_data:
            self.section_data[k] = self.section_data[k].sample(frac=1).reset_index(drop=True)

        print('\nTraining chapter classifier')
        
        self.pipeline = Pipeline([
                        ('vectorizer', TfidfVectorizer(sublinear_tf=True, 
                                                       ngram_range=(1, 2),
                                                       stop_words='english', 
                                                       strip_accents='unicode', 
                                                       decode_error="ignore")),
                        ('classifier', MultinomialNB(alpha=.01))])

        ############# Training the chapter classifier #############
        X = self.data['text'].values
        Y = self.data['class'].values

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
        self.pipeline.fit(x_train, y_train)

        y_real, y_pred = y_test, self.pipeline.predict(x_test)
        print('Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred) * 100))

        ############# Training each section classifier #############
        self.section_pipelines = {}
        for k in sorted(self.section_data.keys()):
            print('Training [ {} ] section classifier'.format(k))
            self.section_pipelines[k] = Pipeline([
                                        ('vectorizer', TfidfVectorizer(sublinear_tf=True,
                                                                       max_df=0.5,
                                                                       ngram_range=(1, 2),
                                                                       stop_words='english', 
                                                                       strip_accents='unicode', 
                                                                       decode_error="ignore")),
                                        ('classifier', MultinomialNB(alpha=.01))])
            X = self.section_data[k]['text'].values
            Y = self.section_data[k]['class'].values
            
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
            self.section_pipelines[k].fit(x_train, y_train)

            y_real, y_pred = y_test, self.section_pipelines[k].predict(x_test)
            print('Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred) * 100))

        ############# Assessing section difficulties ################
        self.section_difficulty = { k : [] for k in self.section_map }
        for x, y in zip(X_questions, Y_questions):
            for sentence in nltk.sent_tokenize(x):
                x_t = self.__preprocess(sentence, remove_stopwords=True)
                chapter = self.pipeline.predict([x_t])[0]
                topic = self.section_pipelines[chapter].predict([x_t])[0]
                if topic != y:
                    chap_num = self.section_map[topic]
                    all_chap_topics = sorted([ k for k in self.section_map if self.section_map[k] == chap_num])
                    probs = self.section_pipelines[chapter].predict_proba([x_t])[0]
                    probs_dict = probs_to_classes_dict(probs, all_chap_topics)
                    nsq_val = probs_dict[topic] # get the right n-squared value for that question from the dict lookup
                else:
                    nsq_val = max(self.section_pipelines[chapter].predict_proba([x_t])[0])

                self.section_difficulty[y].append(nsq_val)

        for k in self.section_difficulty:
            t = self.section_difficulty[k]
            try:
                self.section_difficulty[k] = sum(t) / len(t) 
            except ZeroDivisionError:
                self.section_difficulty[k] = 1

    def classify(self, data):
        if type(data) != type([]):
            data = [data]

        results = []
        for x in data:
            x_t = self.__preprocess(x, remove_stopwords=True)
            chapter = self.pipeline.predict([x_t])[0]
            topic = self.section_pipelines[chapter].predict([x_t])[0]
            nsq_val = max(self.section_pipelines[chapter].predict_proba([x_t])[0]) * self.section_difficulty[topic]
            results.append((topic, nsq_val))

        return results

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

    all_classes = sorted(list(set(list(classifier.data['class'].values))))

    questions = ["""Having some problems implementing a quicksort sorting algorithm in java. I get a stackoverflow error when I run this program and I'm not exactly sure why. If anyone can point out the error, it would be great.""",
    """Give an example that shows that the approximation sequence of Newton's method may diverge. """,
    """Find the number of comparisons made by the sentinel version of linear search  b in the worst case. in the average case if the probability of a successful search is p (0 p 1).""",
    """ Write a brute force pattern matching program for playing the game Battleship on the computer."""]
    
    for i, q in enumerate(questions):
        c, p = classifier.classify(q)[0]
        print('Question', i + 1)
        print(c, '[{}]'.format(p))