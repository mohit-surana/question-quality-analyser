import csv
import keras.backend as K
from keras.models import Sequential
from keras.layers import *
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from gensim.models import Word2Vec
from utils import get_data_for_cognitive_classifiers

import numpy as np
import pickle

import sys

sys.setrecursionlimit(2 * 10 ** 7)

NUM_CLASSES = 6

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K' 

from utils import clean_no_stopwords


with open("models/glove.840B.300d.txt", "r", encoding='utf-8') as lines:
    w2v = {}
  
    for row, line in enumerate(lines):
        try:
            w = line.split()[0]
            vec = np.array(list(map(float, line.split()[1:])))
            w2v[w] = vec
        except:
            continue
        finally:
            print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} GloVe vectors'.format(row + 1))

def sent_to_glove(questions):
	questions_w2glove = []

	for question in questions:
		vec = []
		for word in question[:10]:
			if word in w2v:
				vec.append(w2v[word])
			else:
				vec.append(np.zeros(len(w2v['the'])))
		questions_w2glove.append(np.array(vec))

	return np.array(questions_w2glove)

class SkillClassifier:
	def __init__(self, input_dim=100, hidden_dim=32, dropout=0.2):
		np.random.seed(7)
		
		encoder_a = Sequential()
		encoder_a.add(LSTM(hidden_dim, input_shape=(None, input_dim), recurrent_dropout=dropout))

		encoder_b = Sequential()
		encoder_b.add(LSTM(hidden_dim, input_shape=(None, input_dim), dropout=dropout))

		self.model = Sequential()
		self.model.add(Merge([encoder_a, encoder_b], mode='concat'))
		self.model.add(Dense(NUM_CLASSES, kernel_initializer="lecun_uniform", activation='softmax'))

		self.model.compile(loss='categorical_crossentropy',
		                optimizer='adam',
		                metrics=['accuracy'])
		
	def train(self, X1_train, X2_train, Y_train, X1_val, X2_val, Y_val, epochs=5, batch_size=32):
		print(self.model.summary())
		self.model.fit([X1_train, X2_train], Y_train, epochs=epochs, shuffle=True, batch_size=batch_size) #validation_data=([X1_val, X2_val], Y_val))

	def test(self, X1_test, X2_test, Y_test):
		return self.model.evaluate([X1_test, X2_test], Y_test, verbose=0)[1]

	def predict(self, question):
		question = clean_no_stopwords(question)
		return self.model.predict(sent_to_glove(question), verbose=0)

	def predict_group(self, questions):
		questions = [clean_no_stopwords(q) for q in questions]
		predictions = self.model.predict(questions, batch_size=32, verbose=0)
		
		return {0: [questions[i] for i in range(len(questions)) if not predictions[i] ], 1: [questions[i] for i in range(len(questions)) if predictions[i]]}
		
	def save(self):
		self.model.save('models/rnn_model.h5')

if __name__ == "__main__":
	X_data = []
	Y_data = []

	clf = SkillClassifier(input_dim=len(w2v['the']))

	X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers(threshold=0.20, what_type='ada', split=0.8, include_keywords=True, keep_dup=False)

	X_train = sent_to_glove(X_train)
	X_train = sequence.pad_sequences(X_train, maxlen=10)
	
	for i in range(len(Y_train)):
		v = np.zeros(NUM_CLASSES)
		v[Y_train[i]] = 1
		Y_train[i] = v

	Y_train = np.array(Y_train)

	X_test = sent_to_glove(X_test)
	X_test = sequence.pad_sequences(X_test, maxlen=10)
	
	for i in range(len(Y_test)):
		v = np.zeros(NUM_CLASSES)
		v[Y_test[i]] = 1
		Y_test[i] = v

	Y_test = np.array(Y_test)

	clf.train(X_train, X_train, Y_train, None, None, None, epochs=50)
	print(str(clf.test(X_test, X_test, Y_test) * 100) + '%')

	clf.save()

