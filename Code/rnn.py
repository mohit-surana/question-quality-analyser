import csv
import keras.backend as K
from keras.models import Sequential
from keras.layers import LSTM, Input
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from gensim.models import Word2Vec

import numpy as np
import pickle

import sys

sys.setrecursionlimit(2 * 10 ** 7)

NUM_CLASSES = 6

from utils import clean_no_stopwords

class SkillClassifier:
	def __init__(self, input_dim=50, hidden_dim=32, dropout=0.3):
		np.random.seed(7)
		
		encoder_a = Sequential()
		encoder_a.add(LSTM(hidden_dim, input_shape=(None, input_dim), dropout_W=dropout))

		encoder_b = Sequential()
		encoder_b.add(LSTM(hidden_dim, input_shape=(None, input_dim), dropout_U=dropout))
		self.model = Sequential()
		self.model.add(Merge([encoder_a, encoder_b], mode='concat'))
		
		self.model.add(Dense(NUM_CLASSES, init="lecun_uniform", activation='softmax'))

		self.model.compile(loss='categorical_crossentropy',
		                optimizer='adam',
		                metrics=['accuracy'])
		
		'''
		I = Input(shape=(None, input_dim)) # unknown timespan, fixed feature size
		lstm = LSTM(hidden_dim)
		self.model = K.function(inputs=[I], outputs=[lstm(I)])

		data1 = np.random.random(size=(1, 100, 200)) # batch_size = 1, timespan = 100
		print f([data1])[0].shape
		# (1, 20)

		data2 = np.random.random(size=(1, 314, 200)) # batch_size = 1, timespan = 314
		print f([data2])[0].shape
		# (1, 20)
		'''

	def train(self, X1_train, X2_train, Y_train, X1_val, X2_val, Y_val, epochs=5, batch_size=32):
		print(self.model.summary())
		self.model.fit([X1_train, X2_train], Y_train, nb_epoch=epochs, shuffle=True, batch_size=batch_size, validation_data=([X1_val, X2_val], Y_val))

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

__model = None
def sent_to_glove(question):
	global __model

	if not __model:
		__model = # load model

	'''
	word_list = question.split(" ")
	glove_list = []
	for word in word_list:
		try:
			glove_list.append(__model[word])
		except:
			pass
	'''
	return None #np.array(glove_list)

if __name__ == "__main__":
	X_data = []
	Y_data = []

	clf = SkillClassifier()

	''' 
	# fill X_data and Y_data
	'''



	X_train = np.array(X_data[: int(len(X_data) * 0.70) ])
	Y_train = np.array(Y_data[: int(len(X_data) * 0.70) ])


	X_val = np.array(X_data[int(len(X_data) * 0.70) : int(len(X_data) * 0.9)])
	Y_val = np.array(Y_data[int(len(X_data) * 0.70) : int(len(X_data) * 0.9)])

	X_test = np.array(X_data[int(len(X_data) * 0.9) :])
	Y_test = np.array(Y_data[int(len(X_data) * 0.9) :])

	X_train = sequence.pad_sequences(X_train, maxlen=20)
	X_val = sequence.pad_sequences(X_val, maxlen=20)
	X_test = sequence.pad_sequences(X_test, maxlen=20)

	clf.train(X_train, X_train, Y_train, X_val, X_val, Y_val, epochs=5)
	print str(clf.test(X_test, X_test, Y_test) * 100) + '%'

	clf.save()

