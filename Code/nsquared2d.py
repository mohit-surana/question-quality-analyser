import codecs
import csv
import utils
import numpy as np
import pprint
import classifier as Nsq
from classifier import DocumentClassifier

from sklearn import datasets, svm, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
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
        questions.append(utils.clean2(row[0]))
        labels.append(int(row[1]))

all_classes = sorted(list(set(list(classifier.data['class'].values))))

probs = [max(p) for p in classifier.classify(questions)]

X = np.array([[p] for p in probs])
y = np.array(labels)

data = list(zip(probs, labels, questions))
pprint.pprint(data)
data = sorted(data, key=lambda x: x[0])
rdict = {}
for d in data:
    if d[1] not in rdict.keys():
        rdict[d[1]] = []
    rdict[d[1]].append(d[0])

for k, v in rdict.items():
    rdict[k] = [float("%.2f" %min(v)), 
                float("%.2f" %(np.sum(v).astype(np.float32) / len(v))), 
                float("%.2f" %max(v))]

pprint.pprint(rdict)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


et = GaussianNB()
et.fit(X_train, y_train)

print('Prediction on test')
preds = et.predict(X_test)
print(preds)

print('Original samples')
print(y_test)

print('Accuracy')
print(accuracy_score(y_test, preds))
'''

C_start, C_end, C_step = -3, 15, 2
parameters = {'C': 2. ** np.arange(C_start, C_end + C_step, C_step)}

clf = svm.LinearSVC()
grid = GridSearchCV(clf, parameters).fit(X_train, y_train)
print(grid.score(X_test, y_test))

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

predicted = grid.predict(X_test)
print(y_test)
print(predicted)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)
'''
