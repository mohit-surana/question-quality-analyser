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
from sklearn import linear_model

questions = []
probs = []
labels = []
labels_nsq = []
labels_lda = []
labels_lsa = []
labels_know = []
labels_cog = []
y_combined = []
x_combined = []
knowledge_dim = ['Factual', 'Conceptual', 'Procedural', 'Metacognitive']


def get_model(label):
    return np.array([[p] for p in label]).astype(np.float)
    

def logRegression(probs, y):
    
    global y_combined
    global x_combined
    X = probs
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    x_combined = X_test
    
    logreg = linear_model.LogisticRegression(C=1e5)

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X_train, y_train)
    joblib.dump(logreg, 'models/Knowledge_%s/Logreg/model.pkl' %TEST)
    preds = logreg.predict(X_test)
    print(preds)

    print('Original samples')
    print(y_test)
    y_combined.append(preds)

    print('Accuracy')
    print(accuracy_score(y_test, preds))

def get_prob(probs, y, use='Gaussian'):
    
    global y_combined
    global x_combined
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
    x_combined = X_test
    if(use == 'Gaussian'):
        if LOAD_MODEL:
            et = joblib.load('models/Knowledge_%s/Gaussian/model_50.pkl' %TEST)
        else:
            et = GaussianNB()
            et.fit(X_train, y_train)
            
            joblib.dump(et, 'models/Knowledge_%s/Gaussian/model.pkl' %TEST)
            
        print('Prediction on test')
        preds = et.predict(X_test)
        print(preds)

        print('Original samples')
        print(y_test)
        y_combined.append(preds)

        print('Accuracy')
        print(accuracy_score(y_test, preds))
        
        
    if(use == 'Linear'):
        C_start, C_end, C_step = -3, 15, 2
        parameters = {'C': 2. ** np.arange(C_start, C_end + C_step, C_step)}
        if LOAD_MODEL:
            grid = joblib.load('models/Knowledge_%s/Linear/model_48.pkl' %TEST)
        else:
            clf = svm.SVC(kernel='linear')
            grid = GridSearchCV(clf, parameters).fit(X_train, y_train)

            print("The best parameters are %s with a score of %0.2f"
                  % (grid.best_params_, grid.best_score_))
            
            joblib.dump(grid, 'models/Knowledge_%s/Linear/model.pkl' %TEST)
        print('Prediction on test')
        preds = grid.predict(X_test)
        print(preds)

        print('Original samples')
        print(y_test)
        y_combined.append(preds)
        

        print('Accuracy')
        print(accuracy_score(y_test, preds))
    
    if(use == 'Poly'):
        parameters = {'kernel': ['poly'], 'C': [0.1, 0.5, 1, 10], 'gamma': [0.001, 0.0001], 'degree': [2, 3], 'coef0': [0.0, 0.1]}
        if LOAD_MODEL:
            grid = joblib.load('models/Knowledge_%s/Poly/model.pkl' %TEST)
        else:
            clf = svm.SVC(kernel='poly')
            grid = GridSearchCV(clf, parameters, verbose = 1).fit(X_train, y_train)

            print("The best parameters are %s with a score of %0.2f"
                  % (grid.best_params_, grid.best_score_))
            
            joblib.dump(grid, 'models/Knowledge_%s/Poly/model.pkl' %TEST)
        print('Prediction on test')
        preds = grid.predict(X_test)
        print(preds)
        y_combined.append(preds)

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    if LOAD_MODEL:
        grid = joblib.load('models/Knowledge_%s/Rbf/model_48.pkl' %TEST)
    else:
        clf = svm.SVC(kernel='rbf') #, max_iter=1000)
        parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}
                         
        grid = GridSearchCV(clf, parameters).fit(X_train, y_train)
        
        #print(grid.score(X_test, y_test))

        print("The best parameters are %s with a score of %0.2f"
                  % (grid.best_params_, grid.best_score_))

        joblib.dump(grid, 'models/Knowledge_%s/Rbf/model.pkl' %TEST)
    print('Prediction on test')
    preds = grid.predict(X_test)
    print(preds)
    global y_combined
    y_combined.append(preds)

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
LOAD_MODEL = False

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
                questions.append(row[0])
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
    
    print('\n\nTest on Knowledge domain with n-squared : LINEAR')
    get_prob(get_model(labels_nsq), labels_know, use='Linear')
        
    print('\n\nTest on Knowledge domain with n-squared : POLYNOMIAL')
    get_prob(get_model(labels_nsq), labels_know, use='Poly')

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
    with open('datasets/OS_Exercise_Questions_Relabelled.csv', encoding="latin-1") as csvfile:
        csvreader = csv.reader(csvfile.read().splitlines()[1:])
        for row in csvreader:
            lab = (int(row[1]) - int(row[5]))/6
            questions.append(row[0])
            labels.append(int(row[1]))
            labels_nsq.append(row[2])
            labels_lda.append(row[3])
            labels_lsa.append(row[4])
            labels_cog.append(row[5])
            labels_know.append(lab)
            '''
            if(lab == 0):
                c_0 += 1
                if(c_0 < 120):
                    questions.append(row[0])
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            elif(lab == 1):
                c_1 += 1
                if(c_1 < 120):
                    questions.append(row[0])
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            else:
                questions.append(row[0])
                labels.append(int(row[1]))
                labels_nsq.append(row[2])
                labels_lda.append(row[3])
                labels_lsa.append(row[4])
                labels_cog.append(row[5])
                labels_know.append(lab)
            '''
            
    print('\n\nTest on Knowledge domain with n-squared : Log Reg')
    logRegression(get_model(labels_nsq), labels_know)
    
    print('\n\nTest on Knowledge domain with n-squared : GAUSSIAN')
    get_prob(get_model(labels_nsq), labels_know, use='Gaussian')
    
    print('\n\nTest on Knowledge domain with n-squared : LINEAR')
    get_prob(get_model(labels_nsq), labels_know, use='Linear')
    
    print('\n\nTest on Knowledge domain with n-squared : POLYNOMIAL')
    get_prob(get_model(labels_nsq), labels_know, use='Poly')
    
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
    #global y_combined
    
    ##### WRITE INTO A FILE  ######
    count = 0
    with codecs.open('datasets/OS_Exercise_Questions_Results.csv', 'w', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Questions', 'Manual Label', 'Log Reg', 'Gaussian', 'Linear', 'RBF', 'Polynomial'])
        for question, label_know in zip(questions[-1*len(x_combined):], labels_know):
            csvwriter.writerow([question, knowledge_dim[int(label_know)],
                            knowledge_dim[int(y_combined[0][count])],
                            knowledge_dim[int(y_combined[1][count])], knowledge_dim[int(y_combined[2][count])], knowledge_dim[int(y_combined[4][count])],
                            knowledge_dim[int(y_combined[3][count])] ])
            count += 1


if TEST == 'Combined':
    lab = 0
    c_0 = 0
    c_1 = 0
    c_2 = 0
    with open('datasets/OS_Exercise_Questions_Relabelled.csv', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile.read().splitlines()[1:])
        for row in csvreader:
            lab = (int(row[1]) - int(row[5]))/6
            if(lab == 0):
                c_0 += 1
            questions.append(row[0])
            labels.append(int(row[1]))
            labels_nsq.append(row[2])
            labels_lda.append(row[3])
            labels_lsa.append(row[4])
            labels_cog.append(row[5])
            labels_know.append(lab)
            '''
            if(lab == 0):
                c_0 += 1
                if(c_0 < 120):
                    questions.append(row[0])
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            elif(lab == 1):
                c_1 += 1
                if(c_1 < 120):
                    questions.append(row[0])
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            elif(lab == 2):
                c_2 += 1
                if(c_2 < 120):
                    questions.append(row[0])
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            else:
                questions.append(row[0])
                labels.append(int(row[1]))
                labels_nsq.append(row[2])
                labels_lda.append(row[3])
                labels_lsa.append(row[4])
                labels_cog.append(row[5])
                labels_know.append(lab)
    '''
    print('count of 0s for OS questions are:', c_0)
    c_0 = 0
    c_1 = 0
    c_2 = 0
    with open('datasets/ADA_Exercise_Questions_Relabelled_v5.csv', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile.read().splitlines()[1:])
        for row in csvreader:
            lab = (int(row[1]) - int(row[5]))/6
            questions.append(row[0])
            labels.append(int(row[1]))
            labels_nsq.append(row[2])
            labels_lda.append(row[3])
            labels_lsa.append(row[4])
            labels_cog.append(row[5])
            labels_know.append(lab)
            '''
            if(lab == 0):
                c_0 += 1
                if(c_0 < 30):
                    questions.append(row[0])
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            elif(lab == 1):
                c_1 += 1
                if(c_1 < 30):
                    questions.append(row[0])
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            elif(lab == 2):
                c_2 += 1
                if(c_2 < 30):
                    questions.append(row[0])
                    labels.append(int(row[1]))
                    labels_nsq.append(row[2])
                    labels_lda.append(row[3])
                    labels_lsa.append(row[4])
                    labels_cog.append(row[5])
                    labels_know.append(lab)
            else:
                questions.append(row[0])
                labels.append(int(row[1]))
                labels_nsq.append(row[2])
                labels_lda.append(row[3])
                labels_lsa.append(row[4])
                labels_cog.append(row[5])
                labels_know.append(lab)
            '''
    
    print('\n\nTest on Knowledge domain with n-squared : Log Reg')
    logRegression(get_model(labels_nsq), labels_know)
    
    print('\n\nTest on Knowledge domain with n-squared : GAUSSIAN')
    get_prob(get_model(labels_nsq), labels_know, use='Gaussian')
    
    print('\n\nTest on Knowledge domain with n-squared : LINEAR')
    get_prob(get_model(labels_nsq), labels_know, use='Linear')
    
    print('\n\nTest on Knowledge domain with n-squared : POLYNOMIAL')
    get_prob(get_model(labels_nsq), labels_know, use='Poly')
    
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
    #global y_combined
    
    ##### WRITE INTO A FILE  ######
    count = 0
    #print(len(questions))
    #print(len(questions)//4)
    with codecs.open('datasets/COMBINED_Exercise_Questions_Results.csv', 'w', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Questions', 'Manual Label', 'Log Reg', 'Gaussian', 'Linear', 'RBF', 'Polynomial'])
        for question, label_know in zip(questions[-110:], labels_know):
            csvwriter.writerow([question, knowledge_dim[int(label_know)],
                            knowledge_dim[int(y_combined[0][count])],
                            knowledge_dim[int(y_combined[1][count])], knowledge_dim[int(y_combined[2][count])], knowledge_dim[int(y_combined[4][count])],
                            knowledge_dim[int(y_combined[3][count])] ])
            count += 1

    
