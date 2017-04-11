###### NEURAL NETWORK BASED VOTING SYSTEM ########
import codecs
import csv
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def vote(logreg, gaussian, linearsvc, Y1):
    X = np.array(list(zip(logreg, gaussian, linearsvc)))
    Y = np.array(Y1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(32, 16), batch_size=16, learning_rate='adaptive', learning_rate_init=0.001, verbose=True)
    clf.fit(x_train, y_train)
    print('ANN training completed')
    y_real, y_pred = y_test, clf.predict(x_test)

    joblib.dump(clf, 'models/know_ann_voter.pkl')

    print('Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred) * 100))
    '''
    print('MaxEnt Accuracy: {:.2f}%'.format(accuracy_score(y_real, x_test.T[0]) * 100))
    print('BiRNN Accuracy: {:.2f}%'.format(accuracy_score(y_real, x_test.T[1]) * 100))
    print('SVM-GloVe Accuracy: {:.2f}%'.format(accuracy_score(y_real, x_test.T[2]) * 100))
    '''
know_labels = {'Factual':0, 'Conceptual':1, 'Procedural':2, 'Metacognitive':3}
if __name__ == '__main__':
    questions = []
    logreg = []
    gaussian = []
    linearsvc = []
    Y1 = []
    
    with open('datasets/COMBINED_Exercise_Questions_Results1.csv', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile.read().splitlines()[1:])
        for row in csvreader:
            questions.append(row[0])
            Y1.append(know_labels.get(row[1]))
            logreg.append(know_labels.get(row[2]))
            gaussian.append(know_labels.get(row[3]))
            linearsvc.append(know_labels.get(row[4]))
            
            
    vote(logreg, gaussian, linearsvc, Y1)