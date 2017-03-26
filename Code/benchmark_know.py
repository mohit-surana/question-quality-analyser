import csv
import pickle
import numpy as np

import classifier
from classifier import DocumentClassifier, get_knowledge_probs
from docsim_lsa import clean

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'  
knowledge_mapping = {'Metacognitive': 3, 'Procedural': 2, 'Conceptual': 1, 'Factual': 0}
subject = 'ADA'

questions = []
with open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue

        questions.append([row[0], row[-1]])

x_vals = [k[0] for k in questions]
y_vals = [knowledge_mapping[k[1].split('/')[0]] for k in questions]


''' Knowledge prediction for N-squared'''
classifier_nsq = pickle.load(open('models/%s/__Classifier.pkl' % (subject, ), 'rb'))
correct_nsq = 0
print()
for i, x in enumerate(x_vals):
    print(CURSOR_UP_ONE + ERASE_LINE + '[N-Squared] Testing For', i)
    y_pred = np.argmax(get_knowledge_probs(questions[0], subject))
    if y_pred == y_vals[i]:
        correct_nsq += 1

print('[N-squared] Accuracy: {:.3f}%'.format(float(correct_nsq) / float(len(x_vals)) * 100.0))


lsi_tfidf = models.LsiModel.load('models/lsa/%s_lsi.model' %subject)
corpus_lsi_tfidf = lsi_tfidf[corpus_tfidf]
index = similarities.MatrixSimilarity(corpus_lsi_tfidf, num_features=lsi_tfidf.num_topics)

correct_lsa = 0
print()
for i, x in enumerate(x_vals):
    print(CURSOR_UP_ONE + ERASE_LINE + 'Testing For', i)
    y_pred = np.argmax(get_knowledge_probs(questions[0], subject))
    if y_pred == y_vals[i]:
        correct_nsq += 1

print('N-squared Accuracy: {:.3f}%'.format(float(correct_nsq) / float(len(x_vals)) * 100.0))

query_bow = id2word.doc2bow(clean(query).lower().split())
query_lsi = lsi_tfidf[query_bow]

sims = index[query_lsi]