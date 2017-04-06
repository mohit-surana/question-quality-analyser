import codecs
import csv
import utils
import numpy as np
import pprint
import nsquared as Nsq
from nsquared import DocumentClassifier
from sklearn.externals import joblib
from sklearn import datasets, svm, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

questions = []
probs = []
labels = []
labels_nsq = []
labels_lda = []
labels_lsa = []
labels_know = []
labels_cog = []

def get_model(label):
    return np.array([[p] for p in label]).astype(np.float)

def get_prob(probs, y, use='Linear'):        
    X = probs

    y = np.array(y)
    data = list(zip(probs, labels, questions))
    #pprint.pprint(data)
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

    #pprint.pprint(rdict)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    if(use == 'Gaussian'):
        et = GaussianNB()
        et.fit(X_train, y_train)
        
        joblib.dump(et, 'models\Knowledge_%s\Gaussian\model.pkl' %TEST) 
        
        print('Prediction on test')
        preds = et.predict(X_test)
        print(preds)

        print('Original samples')
        print(y_test)

        print('Accuracy')
        print(accuracy_score(y_test, preds))
        
        
    if(use == 'Linear'):
        C_start, C_end, C_step = -3, 15, 2
        parameters = {'C': 2. ** np.arange(C_start, C_end + C_step, C_step)}

        clf = svm.LinearSVC()
        grid = GridSearchCV(clf, parameters).fit(X_train, y_train)

        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        
        joblib.dump(grid, 'models\Knowledge_%s\Linear\model.pkl' %TEST) 
        print('Prediction on test')
        preds = grid.predict(X_test)
        print(preds)

        print('Original samples')
        print(y_test)

        print('Accuracy')
        print(accuracy_score(y_test, preds))

        #cnf_matrix = confusion_matrix(y_test, preds)
        #print(cnf_matrix)
    


def get_prob_rbf_svm(probs, y):
    X = probs

    y = np.array(y)
    data = list(zip(probs, labels, questions))
    #pprint.pprint(data)
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

    #pprint.pprint(rdict)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    clf = svm.SVC(kernel='rbf') #, max_iter=1000)
    parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}
                     
    grid = GridSearchCV(clf, parameters).fit(X_train, y_train)
    
    #print(grid.score(X_test, y_test))

    print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

    joblib.dump(grid, 'models\Knowledge_%s\Rbf\model.pkl' %TEST) 
    print('Prediction on test')
    preds = grid.predict(X_test)
    print(preds)

    print('Original samples')
    print(y_test)

    print('Accuracy')
    print(accuracy_score(y_test, preds))

count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
#####################  SET TEST SUBJECT  ##########################

TEST = 'OS'

#####################  SET TEST SUBJECT  ##########################
if TEST == 'ADA':
    with open('datasets/ADA_Exercise_Questions_Relabelled_v3.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if(row[5] == '2'):
                count_2 += 1
            if(row[5] == '1'):
                count_1 += 1
            if(row[5] == '0'):
                count_0 += 1
            if(row[5] == '3'):
                count_3 += 1
            
            if((row[5] == '1' and count_1 < 20) or (row[5] == '2' and count_2 < 20) or (row[5] not in ['1', '2'])):
                questions.append(utils.clean2(row[0]))
                labels.append(int(row[1]))
                labels_nsq.append(row[2])
                labels_lda.append(row[3])
                labels_lsa.append(row[4])
                labels_know.append(row[5])
                labels_cog.append(row[6])
    print(count_0, count_1, count_2, count_3)

    '''
    print('Test on N-Squared')
    get_prob(get_model(labels_nsq), labels)

    print('\n\nTest on LDA')
    get_prob(get_model(labels_lda), labels)

    print('\n\nTest on LSA')
    get_prob(get_model(labels_lsa), labels)
    '''

    print('\n\nTest on Knowledge domain with n-squared : GAUSSIAN')
    get_prob(get_model(labels_nsq), labels_know, use='Gaussian')
    print('\n\nTest on Knowledge domain with n-squared : SVM RBF')
    get_prob_rbf_svm(get_model(labels_nsq), labels_know)

    '''
    print('\n\nTest on Cognitive domain with n-squared : GAUSSIAN')
    get_prob_gaussian(get_model(labels_nsq), labels_cog)
    print('\n\nTest on Cognitive domain with n-squared : SVM RBF')
    get_prob_rbf_svm(get_model(labels_nsq), labels_cog)
    '''
if TEST == 'OS':
    lab = 0
    c_0 = 0
    c_1 = 0
    with open('datasets/OS_Exercise_Questions_Relabelled.csv', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile.read().splitlines()[1:])
        for row in csvreader:
            lab = (int(row[1]) - int(row[5]))/6
            if(lab == 0):
                c_0 += 1
                if(c_0 < 120):
                    questions.append(utils.clean2(row[0]))
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            elif(lab == 1):
                c_1 += 1
                if(c_1 < 120):
                    questions.append(utils.clean2(row[0]))
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            else:
                questions.append(utils.clean2(row[0]))
                labels.append(int(row[1]))
                labels_nsq.append(row[2])
                labels_lda.append(row[3])
                labels_lsa.append(row[4])
                labels_cog.append(row[5])
                labels_know.append(lab)

    print('\n\nTest on Knowledge domain with n-squared : GAUSSIAN')
    get_prob(get_model(labels_nsq), labels_know, use='Gaussian')
    
    print('\n\nTest on Knowledge domain with n-squared : LINEAR')
    get_prob(get_model(labels_nsq), labels_know, use='Linear')
    
    print('\n\nTest on Knowledge domain with n-squared : SVM RBF')
    get_prob_rbf_svm(get_model(labels_nsq), labels_know)

    for lab in labels_know:
        if lab == 0:
            count_0 += 1
        elif lab == 1:
            count_1 += 1
        elif lab == 2:
            count_2 += 1
        else:
            count_3 += 1
    print('0s are:', count_0, '1s are:', count_1, '2s are:', count_2, '3s are:', count_3)