from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os 
from utils import clean_no_stopwords, get_data_for_cognitive_classifiers

TRAIN = True

class MNBC(BaseEstimator, ClassifierMixin):
	def __init__(self, tfidf_ngram_range=(1, 2), mnbc_alpha=.05):
		self.clf = Pipeline([ ('vectorizer', TfidfVectorizer(sublinear_tf=True,
                                                    ngram_range=tfidf_ngram_range,
                                                    stop_words='english',
                                                    strip_accents='unicode',
                                                    decode_error="ignore")),
                        ('classifier', MultinomialNB(alpha=mnbc_alpha))])

	def transform(self, X):
		if type(X[0]) == type([]):
			return [' '.join(x) for x in X]
		
		return X

	def fit(self, X, y):
		self.clf.fit(self.transform(X), y)

	def predict(self, X):
		return self.clf.predict(self.transform(X))

	def predict_proba(self, X):
		return self.clf.predict_proba(self.transform(X))

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.3, 0.4, 0.5, 0.6, 0.7], 
																		  what_type=['ada', 'os'], 
																		  split=0.8, 
																		  include_keywords=False, 
																		  keep_dup=False)

	X2_train, Y2_train, X2_test, Y2_test = get_data_for_cognitive_classifiers(threshold=[1], 
																		  what_type=['bcl'], 
																		  split=0.8, 
																		  include_keywords=False, 
																		  keep_dup=False)

	X_train = X_train + X2_train
	Y_train = Y_train + Y2_train
	X_test = X_test + X2_test
	Y_test = Y_test + Y2_test

	X = X_train + X_test
	Y = Y_train + Y_test

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


	print('Loaded/Preprocessed data')

	if TRAIN:
	    clf = MNBC(tfidf_ngram_range=(1, 2), mnbc_alpha=.01)
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
