import numpy as np
import codecs
import csv
from utils import clean
import re
import pickle
from sklearn.externals import joblib
from sklearn import svm

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}
X = []
X_trans = []
Y_cog = []
Y_know = []
FILTERED = False
TRAIN_SVM_GLOVE = False

filtered_suffix = '_filtered' if FILTERED else ''

def train(X, Y, model_name = 'svm_model_glove'):
    clf = svm.LinearSVC()  #clf = svm.SVC(kernel='rbf')
    clf.fit(X,Y)
    joblib.dump(clf, 'models/%s%s.pkl' % (model_name, filtered_suffix, ))
    
def test():
    with codecs.open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
        all_rows = csvfile.read().splitlines()[1:]
        csvreader = csv.reader(all_rows[len(all_rows)*7//10:])
        for row in csvreader:
            sentence, label_cog, label_know = row
            m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
            sentence = m.groups()[2]
            label_cog = label_cog.split('/')[0]
            label_know = label_know.split('/')[0]
            clean_sentence = clean(sentence, return_as_list = False)
            X.append(clean_sentence)
            Y_cog.append(mapping_cog[label_cog])
            Y_know.append(mapping_know[label_know])
            X_trans.extend(obj.transform([clean(sentence)]))
            

    get_labels_batch(X) #probs, labels = get_labels_batch(X)
    #print('Accuracy:', svm.get_prediction(labels, Y_cog)*100, '%')
    #for x, xt in zip(X, X_trans):
    #    print(x, xt)    

def get_labels_batch(questions, model_name='svm_model_glove'):
    
    #CHANGE THINGS HERE 
    labels = []
    probabs = []
    for question in questions:
        
        clf = joblib.load('models/%s%s.pkl' % (model_name, filtered_suffix, ))
        #probs = clf.decision_function(X)
        #probs = abs(1 / probs[0])   

        label = clf.predict(question)
        print(label)
        '''
        labels.append(label)
        probs = np.exp(probs) / np.sum(np.exp(probs))
    
        for i in range(label + 1, 6):
            probs[i] = 0.0
        
        probabs.append(probs)

    return probabs, labels
    '''
    
def get_prediction(labels, targets):
    count = 0
    result = list(zip(labels, targets))
    for res in result:
        if res[0] == res[1]:
            count += 1
    return count/len(result)
    
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.items())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

if __name__ == '__main__':            

    with open("models/glove.6B.50d.txt", "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}
        obj = TfidfEmbeddingVectorizer(w2v)
    
    with codecs.open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
        all_rows = csvfile.read().splitlines()[1:]
        csvreader = csv.reader(all_rows[:len(all_rows)*7//10])
        for row in csvreader:
            sentence, label_cog, label_know = row
            m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
            sentence = m.groups()[2]
            label_cog = label_cog.split('/')[0]
            label_know = label_know.split('/')[0]
            clean_sentence = clean(sentence, return_as_list = False)
            X.append(clean_sentence)
            Y_cog.append(mapping_cog[label_cog])
            Y_know.append(mapping_know[label_know])
            X_trans.extend(obj.transform([clean(sentence)]))
    
    with codecs.open('datasets/BCLs_Question_Dataset.csv', 'r', encoding="utf-8") as csvfile:
        all_rows = csvfile.read().splitlines()[1:]
        csvreader = csv.reader(all_rows)  #csvreader = csv.reader(all_rows[:len(all_rows)*7//10])
        for row in csvreader:
            sentence, label_cog = row
            clean_sentence = clean(sentence)
            X.append(clean_sentence)
            Y_cog.append(mapping_cog[label_cog])
            # TODO: Label
            Y_know.append(1)
            X_trans.extend(obj.transform([clean(sentence)]))
            
    domain_keywords = pickle.load(open('resources/domain.pkl', 'rb'))
    for key in domain_keywords:
        for word in domain_keywords[key]:            
            X.append([word])
            Y_cog.append(mapping_cog[key])
            X_trans.extend(obj.transform([clean(word)]))
    
    print('Glove converison done')
    
    if TRAIN_SVM_GLOVE:
        train(X_trans, Y_cog)
    print('Training done')
    print('Testing Started')
    test()
    print('Testing Ended')


