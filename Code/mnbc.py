import os
import random
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from utils import clean_no_stopwords, get_data_for_cognitive_classifiers

TRAIN = True

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}

domain = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/domain.pkl'),  'rb'))

keywords = set()
keyword_doc = ['' for i in range(len(mapping_cog))]
for k in domain:
    for word in domain[k]:
        cleaned_word = clean_no_stopwords(word, lemmatize=False, stem=False, as_list=False)
        keywords.add(cleaned_word)
        keyword_doc[mapping_cog[k]] += cleaned_word + '. '

class MNBC(BaseEstimator, ClassifierMixin):
    def __init__(self, tfidf_ngram_range=(1, 2), mnbc_alpha=.05):
        self.tfidf_ngram_range = tfidf_ngram_range
        self.mnbc_alpha = mnbc_alpha

        tfidf = TfidfVectorizer(norm='l2',
                                min_df=1,
                                decode_error="ignore",
                                use_idf=False,
                                sublinear_tf=True,)

        self.clf = Pipeline([ ('vectorizer', tfidf),
                              ('classifier', MultinomialNB(alpha=self.mnbc_alpha))])

    def __prep_data(self, X):
        if type(X[0]) != str:
            return [' '.join(x) for x in X]
        
        return X

    def fit(self, X, Y):
        docs_train = ['' for i in range(len(mapping_cog))]

        for x, y in zip(X, Y):
            docs_train[y] += ' '.join(x) + '. '

        for i in range(len(docs_train)):
            docs_train[i] += keyword_doc[i]

        tfidf = self.clf.named_steps['vectorizer']

        self.clf.fit(self.__prep_data(X), Y)

    def predict(self, X):
        return self.clf.predict(self.__prep_data(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.__prep_data(X))

if __name__ == '__main__':
    X_train, Y_train = get_data_for_cognitive_classifiers(threshold=[0.20, 0.25],
                                                          what_type=['ada', 'os', 'bcl'],
                                                          include_keywords=True,
                                                          keep_dup=False)
    print(len(X_train))

    X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.25],
                                                        what_type=['ada', 'os', 'bcl'],
                                                        what_for='test',
                                                        keep_dup=False)

    print('Loaded/Preprocessed data')

    if TRAIN:
        clf = MNBC(tfidf_ngram_range=(1, 2), mnbc_alpha=.05)
        clf.fit(X_train, Y_train)
        joblib.dump(clf, os.path.join(os.path.dirname(__file__), 'models/MNBC/mnbc.pkl'))

    else:
        clf = joblib.load(os.path.join(os.path.dirname(__file__), 'models/MNBC/mnbc.pkl'))
    
    Y_true, Y_pred = Y_test, clf.predict(X_test)

    nCorrect = 0
    for i in range(len(Y_true)):
        if Y_true[i] == Y_pred[i]:
            nCorrect += 1

    print()
    print('Accuracy: {:.3f}%'.format(nCorrect / len(Y_test) * 100))

    print(classification_report(Y_true, Y_pred))
