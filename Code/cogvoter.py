import csv
import dill
import re
import nltk
import numpy as np
import pickle
import random
import brnn
import os
from brnn import BiDirectionalRNN, sent_to_glove, clip
from utils import get_filtered_questions, clean_no_stopwords, clean
from sklearn.externals import joblib
from maxent import features
from svm_glove import TfidfEmbeddingVectorizer

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

domain = pickle.load(open('resources/domain.pkl',  'rb'))
domain = { k : set(clean_no_stopwords(' '.join(list(domain[k])), stem=False)) for k in domain.keys() } 
domain_names = domain.keys()

keywords = set()
for k in domain:
	keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_cog2 = { v : k for k, v in mapping_cog.items()}

# transformation for BiRNN. This should actually become a part of the RNN for better code maintainability
INPUT_SIZE = 300
NUM_QUESTIONS = 1000
filename = 'glove.6B.%dd.txt' %INPUT_SIZE

if not os.path.exists('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0]):
	print()
	with open('resources/GloVe/' + filename, "r", encoding='utf-8') as lines:
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
	
	dill.dump(w2v, open('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0], 'wb'))
else:
	w2v = dill.load(open('resources/GloVe/%s_saved.pkl' %filename.split('.txt')[0], 'rb'))

print('Loaded GloVe model')

################# LOADING SO[ADA] questions ################# 

ADA_questions = []
ADA_questions_cleaned = []
with open('datasets/ADA_SO_Questions.csv', 'r', encoding='utf-8') as csvfile:
	print()
	csvreader = csv.reader(csvfile)
	for i, row in enumerate(csvreader):
		if i == 0 or len(row) == 0: 
			continue
		_, sentence, _ = row
		clean_sentence = clean(sentence, return_as_list=False, stem=False)
		if len(set(clean_sentence.split()).intersection(keywords)) and '?' in sentence:
			ADA_questions.append(sentence)
			ADA_questions_cleaned.append(clean_sentence)
			print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} ADA questions'.format(len(ADA_questions)))
			if len(ADA_questions) == NUM_QUESTIONS:
				break

ADA_questions_filtered = get_filtered_questions(ADA_questions_cleaned, threshold=0.25, what_type='ada')
ADA_questions_filtered_for_maxent = get_filtered_questions(ADA_questions_cleaned, threshold=0.75, what_type='ada')

t_ADA = list(zip(ADA_questions, ADA_questions_cleaned, ADA_questions_filtered_for_maxent, ADA_questions_filtered))
random.shuffle(t_ADA)
ADA_questions = [t[0] for t in t_ADA if t[-1].strip() != '']
ADA_questions_cleaned = [t[1] for t in t_ADA if t[-1].strip() != '']
ADA_questions_filtered_for_maxent = [t[2] for t in t_ADA if t[-1].strip() != '']
ADA_questions_filtered = [t[3] for t in t_ADA if t[-1].strip() != '']

################# LOADING SO[OS] questions ################# 

OS_questions = []
OS_questions_cleaned = []
with open('datasets/OS_SO_Questions.csv', 'r', encoding='utf-8') as csvfile:
	print()
	csvreader = csv.reader(csvfile)
	for i, row in enumerate(csvreader):
		if i == 0 or len(row) == 0:
			continue
		_, sentence, _ = row
		clean_sentence = clean(sentence, return_as_list=False, stem=False)
		if len(set(clean_sentence.split()).intersection(keywords)) and '?' in sentence:
			OS_questions.append(sentence)
			OS_questions_cleaned.append(clean_sentence)
			print(CURSOR_UP_ONE + ERASE_LINE + 'Processed {} OS questions'.format(len(OS_questions)))

			if len(OS_questions) == NUM_QUESTIONS:
				break

OS_questions_filtered = get_filtered_questions(OS_questions_cleaned, threshold=0.25, what_type='os')
OS_questions_filtered_for_maxent = get_filtered_questions(OS_questions_cleaned, threshold=0.75, what_type='os')
t_OS = list(zip(OS_questions, OS_questions_cleaned, OS_questions_filtered_for_maxent, OS_questions_filtered))
random.shuffle(t_OS)
OS_questions = [t[0] for t in t_OS if t[-1].strip() != '']
OS_questions_cleaned = [t[1] for t in t_OS if t[-1].strip() != '']
OS_questions_filtered_for_maxent = [t[2] for t in t_OS if t[-1].strip() != '']
OS_questions_filtered = [t[3] for t in t_OS if t[-1].strip() != '']

X_data = ADA_questions_filtered + OS_questions_filtered
X_data_for_maxent = ADA_questions_filtered_for_maxent + OS_questions_filtered_for_maxent
X_data_featureset = [features(X_data_for_maxent[i].split()) for i in range(len(X_data_for_maxent))]
X_data_glove = sent_to_glove(X_data, w2v) 

################# MAXENT MODEL ################# 
clf_maxent = pickle.load(open('models/MaxEnt/maxent_85.pkl', 'rb'))
print('Loaded MaxEnt model')

pred_maxent = []
for x in X_data_featureset:
	pred_maxent.append(clf_maxent.classify(x))
print('MaxEnt classification complete')

################# SVM-GLOVE MODEL #################
clf_svm = joblib.load('models/SVM/glove_svm_model_81.pkl')
print('Loaded SVM-GloVe model')

pred_svm = clf_svm.predict([x.split() for x in  X_data])
print('SVM-GloVe classification complete')

################# BiRNN MODEL #################
clf_brnn = dill.load(open('models/BiRNN/brnn_model_6B-300_72.pkl', 'rb'))
print('Loaded BiRNN model')

pred_brnn = []
for x in X_data_glove:
	pred_brnn.append(clf_brnn.forward(clip(x)))
print('BiRNN classification complete')

################# DUMPING OUTPUT TO CSV #################

with open('datasets/SO_Questions_Cog_Prediction.csv', 'w', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Question', 'Cog(MaxEnt)', 'Cog(BiRNN)', 'Cog(SVM-GloVe)'])
    for q, p_maxent, p_brnn, p_svm in zip(ADA_questions + OS_questions, pred_maxent, pred_brnn, pred_svm):
    	csvwriter.writerow([q, mapping_cog2[p_maxent], mapping_cog2[p_brnn], mapping_cog2[p_svm]])

