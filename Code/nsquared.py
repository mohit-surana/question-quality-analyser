'''
1D prediction for knowledge
hyperparam
toggle para/section
'''

import nltk
import os
import pprint
import re
import pickle
import csv
import platform
import copy

from nltk import stem

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as stp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pandas import DataFrame

from utils import get_questions_by_section

import sys

knowledge_mapping = {'Metacognitive': 3, 'Procedural': 2, 'Conceptual': 1, 'Factual': 0}
knowledge_mapping2 = {v : k for k, v in knowledge_mapping.items()}

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

class DocumentClassifier:
    stemmer = stem.porter.PorterStemmer()
    wordnet = WordNetLemmatizer()

    punkt = {',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '.', '?', '!', '`', '|', '-', '=', '+', '_', '>', '<'}

    if(platform.system() == 'Windows'):
        stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r', encoding='utf8').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
    else:
        stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
    stopwords.update(punkt)

    stopwords2 = stp.words('english')

    def __preprocess(self, text, stop_strength=0):
        text = re.sub('-\n', '', text).lower()
        text = re.sub('\n', ' ', text)
        text = re.sub('[^a-z ]', '', text)
        tokens = [word for word in text.split()]  
        if stop_strength == 0:
            return ' '.join([self.wordnet.lemmatize(i) for i in tokens])
        elif stop_strength == 1:
            return ' '.join([self.wordnet.lemmatize(i) for i in tokens if i not in self.stopwords2])
        else:
            return ' '.join([self.wordnet.lemmatize(i) for i in tokens if i not in self.stopwords])

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

                if title in self.chapter_map.values():
                    chapter = title
                else:
                    chapter = self.chapter_map[self.section_map[content.split('\n')[0]]]

                # training chapters one sentence at a time
                rows_section, rows_chapter = [], []
                for sentence in nltk.sent_tokenize(content):
                    text = self.__preprocess(sentence, stop_strength=1)
                    if text and len(text.split()) > 2:
                        rows_chapter.append({'text' : text, 'class' : chapter })

                    text = self.__preprocess(sentence, stop_strength=2)
                    if text and len(text.split()) > 2:
                        rows_chapter.append({'text' : text, 'class' : chapter })

                # training the sections by passing the entire body of text at once
                if title != chapter:
                    rows_section.append({'text' : self.__preprocess(content, stop_strength=1), 'class' : title })
                    rows_section.append({'text' : self.__preprocess(content, stop_strength=2), 'class' : title })
                self.data = self.data.append(DataFrame(rows_chapter))
                self.section_data[chapter] = self.section_data[chapter].append(DataFrame(rows_section))

                print('Loaded and processed', file_name)

        if subject == 'ADA':
            X_questions, Y_questions = get_questions_by_section(subject, skip_files, shuffle=False)
            '''
            x_qtrain, x_qtest, y_qtrain, y_qtest = train_test_split(X_questions, Y_questions, test_size=0.05)
            rows = []
            for x, y in zip(x_qtrain, y_qtrain):
                chapter = self.chapter_map[self.section_map[y]]
                self.section_data[chapter] = self.section_data[chapter].append(DataFrame([{'text' : self.__preprocess(x, stop_strength=1), 'class' : y}]))
            '''     
        elif subject == 'OS':
            pass # consider training questions on a chapter wise basis to improve accuracy 

        self.data = self.data.sample(frac=1).reset_index(drop=True)
        for k in self.section_data:
            self.section_data[k] = self.section_data[k].sample(frac=1).reset_index(drop=True)

        ############# Training the chapter classifier #############
        print('\nTraining chapter classifier')
        
        self.pipeline = Pipeline([
                        ('vectorizer', TfidfVectorizer(sublinear_tf=True, 
                                                       ngram_range=(1, 2),
                                                       stop_words='english', 
                                                       strip_accents='unicode', 
                                                       decode_error="ignore")),
                        ('classifier', MultinomialNB(alpha=.01))])

        
        X = self.data['text'].values
        Y = self.data['class'].values

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
        self.pipeline.fit(x_train, y_train)

        y_real, y_pred = y_test, self.pipeline.predict(x_test)
        print('Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred) * 100))

        ############# Training each section classifier #############
        self.section_pipelines = {}
        for k in sorted(self.section_data.keys()):
            print('Training [ {} ] classifier'.format(k))
            self.section_pipelines[k] = Pipeline([
                                        ('vectorizer', TfidfVectorizer(sublinear_tf=True,
                                                                       max_df=0.5,
                                                                       ngram_range=(1, 3),
                                                                       stop_words='english', 
                                                                       strip_accents='unicode', 
                                                                       decode_error="ignore")),
                                        ('classifier', MultinomialNB(alpha=.01))])
            X = self.section_data[k]['text'].values
            Y = self.section_data[k]['class'].values

            self.section_pipelines[k].fit(X, Y)

        ############# Assessing section difficulties ################
        if(subject == 'ADA'):
            self.section_difficulty = { k : [] for k in self.section_map }
            nCorrect, nTotal = 0, 0
            for x, y in zip(X_questions, Y_questions):
                for sentence in nltk.sent_tokenize(x):
                    nTotal += 1
                    x_t = self.__preprocess(sentence, stop_strength=1)
                    if len(x_t.split()) < 2:
                        continue
                    chapter = self.pipeline.predict([x_t])[0]
                    topic = self.section_pipelines[chapter].predict([x_t])[0]
                    if topic != y:
                        chap_num = self.section_map[topic]
                        all_chap_topics = sorted([ k for k in self.section_map if self.section_map[k] == chap_num])
                        probs = self.section_pipelines[chapter].predict_proba([x_t])[0]
                        probs_dict = probs_to_classes_dict(probs, all_chap_topics)
                        nsq_val = probs_dict[topic] # get the right n-squared value for that question from the dict lookup
                    else:
                        nCorrect += 1
                        nsq_val = max(self.section_pipelines[chapter].predict_proba([x_t])[0])

                    self.section_difficulty[y].append(nsq_val)

            print('Combined Accuracy: {:.2f}%'.format(nCorrect / nTotal * 100))

            for k in self.section_difficulty:
                t = self.section_difficulty[k]
                try:
                    self.section_difficulty[k] = sum(t) / len(t) 
                except ZeroDivisionError:
                    self.section_difficulty[k] = 1

        else: # os does not have section wise questions, so just keep a default difficulty [alt: apply same difficulty to all sections in a chapter]
            self.section_difficulty = { k : 1 for k in self.section_map }


    def classify(self, data):
        if type(data) != type([]):
            data = [data]

        results = []
        for x in data:
            x_t = self.__preprocess(x, stop_strength=1)
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

    TRAIN = False
    if TRAIN:
        classifier = DocumentClassifier(subject=subject, skip_files={'__', '.DS_Store', 'Key Terms, Review Questions, and Problems', 'Recommended Reading and Web Sites', 'Recommended Reading', 'Summary', 'Exercises', 'Introduction'})
        classifier.save('models/Nsquared/%s/nsquared.pkl' % (subject, ))

    classifier = pickle.load(open('models/Nsquared/%s/nsquared.pkl' % (subject, ), 'rb'))

    all_classes = sorted(list(set(list(classifier.data['class'].values))))

    if subject == 'ADA':
        questions = ["""Having some problems implementing a quicksort sorting algorithm in java. I get a stackoverflow error when I run this program and I'm not exactly sure why. If anyone can point out the error, it would be great.""",
        """Give an example that shows that the approximation sequence of Newton's method may diverge. """,
        """Find the number of comparisons made by the sentinel version of linear search  b in the worst case. in the average case if the probability of a successful search is p (0 p 1).""",
        """ Write a brute force pattern matching program for playing the game Battleship on the computer."""]

    else:
        questions = ['''List and briefly describe some of the defenses against buffer overflows that can be implemented when running existing, vulnerable programs.''',
                    '''What is client/server computing?''',
                    '''What distinguishes client/server computing from any other form of distributed data processing?''',
                    '''What is the role of a communications architecture such as TCP/IP in a client/server environment?''']
    
    for i, q in enumerate(questions):
        c, p = classifier.classify(q)[0]
        print('Question', i + 1)
        print(c, '[{}]'.format(p))