from keras.models import Sequential
import os
import sys

import dill
import numpy as np
from keras.layers import *
from keras.models import Sequential
from keras.preprocessing import sequence

from utils import get_data_for_cognitive_classifiers

sys.setrecursionlimit(2 * 10 ** 7)

NUM_CLASSES = 6

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

INPUT_SIZE = 300
filename = 'glove.840B.%dd.txt' %INPUT_SIZE

if not os.path.exists('models/%s_saved.pkl' %filename.split('.txt')[0]):
    print()
    with open('models/' + filename, "r", encoding='utf-8') as lines:
        w2v = {}
        for row, line in enumerate(lines):
            try:
                w = line.split()[0]
                if w not in vocabulary:
                    continue
                vec = np.array(list(map(float, line.split()[1:])))
                w2v[w] = vec
            except:
                continue
            finally:
                print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} GloVe vectors'.format(row + 1))

    dill.dump(w2v, open('models/%s_saved.pkl' %filename.split('.txt')[0], 'wb'))
else:
    w2v = dill.load(open('models/%s_saved.pkl' %filename.split('.txt')[0], 'rb'))

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
    def __init__(self, input_dim=300, hidden_dim=128, dropout=0.2):
        np.random.seed(7)

        '''
        self.model.add(LSTM(hidden_dims[0], input_shape=(None, input_dim), return_sequences=True, recurrent_dropout=dropout))
        self.model.add(LSTM(hidden_dims[1], dropout=dropout))
        self.model.add(Dense(NUM_CLASSES, kernel_initializer="lecun_uniform", activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])
        '''

        self.model = Sequential()
        self.model.add(LSTM(hidden_dim, input_shape=(None, input_dim), recurrent_dropout=dropout))
        self.model.add(Dense(NUM_CLASSES, kernel_initializer="lecun_uniform", activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

    def train(self, train_data, val_data, epochs=5, batch_size=32):
        print(self.model.summary())
        self.model.fit(train_data[0], train_data[1], epochs=epochs, shuffle=True, batch_size=batch_size, validation_data=(val_data[0], val_data[1]))

    def test(self, test_data):
        return self.model.evaluate(test_data[0], test_data[1], verbose=0)[1]

    def save(self):
        self.model.save('models/rnn_model.h5')

if __name__ == "__main__":
    X_data = []
    Y_data = []

    clf = SkillClassifier(input_dim=len(w2v['the']))

    X_train, Y_train, X_test, Y_test = get_data_for_cognitive_classifiers(threshold=[0.1, 0.15, 0.2, 0.25], what_type=['ada', 'bcl', 'os'], split=0.8, include_keywords=True, keep_dup=False)

    X_data = X_train + X_test
    Y_data = Y_train + Y_test

    X_data = sequence.pad_sequences(sent_to_glove(X_data), maxlen=10)

    for i in range(len(Y_data)):
        v = np.zeros(NUM_CLASSES)
        v[Y_data[i]] = 1
        Y_data[i] = v

    Y_data = np.array(Y_data)

    X_train = np.array(X_data[: int(len(X_data) * 0.70) ])
    Y_train = np.array(Y_data[: int(len(X_data) * 0.70) ])

    X_val = np.array(X_data[int(len(X_data) * 0.70) : int(len(X_data) * 0.8)])
    Y_val = np.array(Y_data[int(len(X_data) * 0.70) : int(len(X_data) * 0.8)])

    X_test = np.array(X_data[int(len(X_data) * 0.8) :])
    Y_test = np.array(Y_data[int(len(X_data) * 0.8) :])

    clf.train(train_data=(X_train, Y_train), val_data=(X_val, Y_val), epochs=10, batch_size=4)
    print(str(clf.test(test_data=(X_test, Y_test)) * 100)[:5] + '%')

    clf.save()

