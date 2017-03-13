import codecs
import csv
import utils
import numpy as np

from classifier import DocumentClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import datasets, svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import confusion_matrix

import pickle

subject = 'ADA'

classifier = pickle.load(open('models/%s/__Classifier.pkl' % (subject, ), 'rb'))

questions = []
probs = []
labels = []
with open('datasets/ADA_Exercise_Questions_Relabelled.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        questions.append(utils.clean(row[0], stem=False, return_as_list=False))
        labels.append(int(row[1]))

probs = [max(p) for p in classifier.classify(questions)]

X = np.array([[p] for p in probs])
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  

parameters = {'kernel': ('linear', 'rbf', 'sigmoid'), 
              'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              'gamma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5]}

grid = GridSearchCV(svm.SVC(), parameters)
print(grid.fit(X_train, y_train).score(X_test, y_test))

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

predicted = grid.predict(X_test)
print(y_test)
print(predicted)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)

