import csv
import dill
import re
import string
import nltk
import numpy as np
import pickle
import platform
import random

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from qfilter_train import tokenizer
from sklearn.externals import joblib

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

SUBJECT_ADA = 'ADA'
SUBJECT_OS = 'OS'

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}

if(platform.system() == 'Windows'):
	stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r', encoding='utf8').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
else:
	stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

regex = re.compile('[%s]' % re.escape(string.punctuation))
#see documentation here: http://docs.python.org/2/library/string.html


########################### PREPROCESSING UTILITY CODE ###########################
def clean(sentence, stem=True, return_as_list=True):
	sentence = sentence.lower()
	final_sentence = []
	for word in word_tokenize(sentence):
		word = regex.sub(u'', word)
		if not (word == u'' or word == ''):
			word = wordnet.lemmatize(word)
			if stem:
				word = porter.stem(word)
			#word = snowball.stem(word)
			final_sentence.append(word)

	return final_sentence if return_as_list else ' '.join(final_sentence)

def clean2(text):
	tokens = [word for word in nltk.word_tokenize(text) if word.lower() not in stopwords]
	return ' '.join(list(set([porter.stem(i) for i in [j for j in tokens if re.match('[a-zA-Z]', j) ]])))

def clean_no_stopwords(text, as_list=True, stem=True):
	tokens = [wordnet.lemmatize(w) for w in text.lower().split() if w.isalpha()]
	if stem:
		tokens = [porter.stem(w) for w in tokens]
	if as_list:
		return tokens
	else:
		return ' '.join(tokens)

########################### SKILL: QUESTION FILTERER ###########################

def get_filtered_questions(questions, threshold=0.25, what_type='os'):
	t_stopwords = set(nltk.corpus.stopwords.words('english'))

	try:
		domain = pickle.load(open('resources/domain.pkl',  'rb'))
	except:
		domain = pickle.load(open('resources/domain_2.pkl',  'rb'))

	keywords = set()
	for k in domain:
		keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))
	t_stopwords = t_stopwords - keywords

	if type(questions) != type([]):
		questions = [questions]

	sklearn_tfidf = pickle.load(open('models/tfidf_filterer_%s.pkl' %what_type.lower(), 'rb'))
	tfidf_matrix = sklearn_tfidf.transform(questions)
	feature_names = sklearn_tfidf.get_feature_names()

	new_questions = []

	for i in range(0, len(questions)):
		feature_index = tfidf_matrix[i,:].nonzero()[1]
		tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
		word_dict = {w : s for w, s in [(feature_names[j], s) for (j, s) in tfidf_scores]}

		question = re.sub(' [^a-z]*? ', ' ', questions[i].lower())
		question = re.split('([.!?])', question)

		sentences = []
		for k in range(len(question) - 1):
			if re.search('[!.?]', question[k + 1]):
				sentences.append(question[k] + question[k + 1])
			elif re.search('[!.?]', question[k]):
				continue
			elif question[k] != '':
				sentences.append(question[k].strip())

		if len(sentences) >= 3:
			q = sentences[0] + ' ' + sentences[-2]
			q += ' ' + sentences[-1] if 'hint' not in sentences[-1] else ''
			questions[i] = q

		new_question = ''
		for word in re.sub('[^a-z ]', '', questions[i].lower()).split():
			try:
				if word.isalpha() and (word_dict[word] < threshold or word in keywords) and word not in t_stopwords:
					new_question += word + ' '
			except:
				pass

		new_questions.append(new_question.strip())

	return new_questions if len(new_questions) > 1 else new_questions[0]

########################### SKILL: GET FILTERED DATA FROM APPROPRIATE DATASETS ###########################

def get_data_for_cognitive_classifiers(threshold=[0, 0.1, 0.15], what_type=['ada', 'os', 'bcl'], split=0.7, include_keywords=True, keep_dup=False, shuffle=True):
	X = []
	Y_cog = []
	Y_know = []
	
	if 'ada'in what_type:
		with open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r', encoding='utf-8') as csvfile:
			X_temp = []
			Y_cog_temp = []
			all_rows = csvfile.read().splitlines()[1:]
			csvreader = csv.reader(all_rows)
			for row in csvreader:
				sentence, label_cog, label_know = row
				m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
				sentence = m.groups()[2]
				label_cog = label_cog.split('/')[0]
				clean_sentence = clean(sentence, return_as_list=False, stem=False)
				X_temp.append(clean_sentence)
				Y_cog_temp.append(mapping_cog[label_cog])

		for t in threshold:
			X_temp_2 = get_filtered_questions(X_temp, threshold=t, what_type='ada')
			X.extend(X_temp_2)
			Y_cog.extend(Y_cog_temp)

	if 'os' in what_type:
		with open('datasets/OS_Exercise_Questions_Labelled.csv', 'r', encoding='utf-8') as csvfile:
			X_temp = []
			Y_cog_temp = []
			all_rows = csvfile.read().splitlines()[5:]
			csvreader = csv.reader(all_rows)
			for row in csvreader:
				shrey_cog, shiva_cog, mohit_cog = row[2].split('/')[0], row[4].split('/')[0], row[6].split('/')[0]
				label_cog = mohit_cog if mohit_cog else (shiva_cog if shiva_cog else shrey_cog)
				label_cog = label_cog.strip()                
				clean_sentence = clean(row[0], return_as_list=False, stem=False)
				X_temp.append(clean_sentence)
				Y_cog_temp.append(mapping_cog[label_cog])

		for t in threshold:
			X_temp_2 = get_filtered_questions(X_temp, threshold=t, what_type='ada')
			X.extend(X_temp_2)
			Y_cog.extend(Y_cog_temp)

	if 'bcl' in what_type:
		with open('datasets/BCLs_Question_Dataset.csv', 'r', encoding='utf-8') as csvfile:
			X_temp = []
			Y_cog_temp = []
			all_rows = csvfile.read().splitlines()[1:]
			csvreader = csv.reader(all_rows)
			for row in csvreader:
				sentence, label_cog = row
				clean_sentence = clean(sentence, return_as_list=False, stem=False)
				X_temp.append(clean_sentence)
				Y_cog_temp.append(mapping_cog[label_cog])

		for t in threshold:
			X_temp_2 = get_filtered_questions(X_temp, threshold=t, what_type='ada')
			X.extend(X_temp_2)
			Y_cog.extend(Y_cog_temp)

	if keep_dup:
		X = [x.split() for x in X]
	else:
		X = [list(np.unique(x.split())) for x in X]
	dataset = list(zip(X, Y_cog))
	
	if shuffle:
		random.shuffle(dataset)

	X_train = []
	Y_train = []
	X_test = []
	Y_test = []

	for x, y in dataset[:int(len(dataset) * split)]:
		if len(x) == 0:
			continue
		X_train.append(x)
		Y_train.append(y)

	for x, y in dataset[int(len(dataset) * split):]:
		if len(x) == 0:
			continue
		X_test.append(x)
		Y_test.append(y)

	if include_keywords:
		domain_keywords = pickle.load(open('resources/domain.pkl', 'rb'))
		for key in domain_keywords:
			for word in domain_keywords[key]:
				X_train.append(clean(word, return_as_list=True, stem=False))
				Y_train.append(mapping_cog[key])

		dataset = list(zip(X_train, Y_train))
		
		if shuffle:
			random.shuffle(dataset)
		
		X_train = [x[0] for x in dataset]
		Y_train = [y[1] for y in dataset]

	return X_train, Y_train, X_test, Y_test

##################### KNOWLEDGE: CONVERT PROB TO HARDCODED VECTOR #####################

def get_knowledge_probs(prob):
	hardcoded_matrix = [1.0, 0.6, 0.3, 0.1, 0]
	for i, r in enumerate(zip(hardcoded_matrix[1:], hardcoded_matrix[:-1])):
		if r[0] <= prob < r[1]:
			break
	level = i

	probs = [0.0] * 4
	for i in range(level):
		# probs[i] = (i + 1) * prob / (level * (level + 1) / 2)
		probs[i] = (i + 1) * prob / (level + 1)
	probs[level] = prob

	return probs

##################### KNOWLEDGE: GET QUESTIONS (relabel.py CODE) #####################

def get_questions_by_section(subject, skip_files, shuffle=True):
	exercise_content = {}
	for filename in sorted(os.listdir('./resources/%s' %subject)):
		with open('./resources/%s/'%subject + filename, encoding='latin-1') as f:
			contents = f.read()
			title = contents.split('\n')[0].strip()
			if len([1 for k in skip_files if (k in title or k in filename)]):
				continue

			match = re.search(r'\n[\s]*Exercises[\s]+([\d]+\.[\d]+)[\s]*(.*)', contents, flags=re.M | re.DOTALL) 

			if match:
				exercise_content[title] = '\n' + match.group(2).split('SUMMARY')[0]

		X_data, Y_data = [], []
		for e in exercise_content:
			for question in re.split('[\n][\s]*[\d]+\.', exercise_content[e].strip(), flags=re.M | re.DOTALL):
				if len(question) > 0:
					X_data.append(re.sub('\n', ' ', re.sub('1\.', '', question.strip()), flags=re.M | re.DOTALL))
					Y_data.append(e)

	if shuffle:
		X = list(zip(X_data, Y_data))
		random.shuffle(X)
		X_data = [x[0] for x in X]
		Y_data = [x[1] for x in X]

	return X_data, Y_data

def get_data_for_knowledge_classifiers(subject='ADA', shuffle=True):
	X_data = []
	Y_data = []
	if subject == SUBJECT_ADA:
		with codecs.open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
			csvreader = csv.reader(csvfile.read().splitlines()[1:])
			for row in csvreader:
				sentence, label_cog, label_know = row
				m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
				X_data.append(m.groups()[2])
				Y_data.append(mapping_know[label_know.split('/')[0]])

	elif subject == SUBJECT_OS: # return question : knowledge mapping
		with codecs.open('datasets/OS_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
			csvreader = csv.reader(csvfile.read().splitlines()[5:])
			for row in csvreader:
				X_data.append(row[0])                
				if(row[5] == '' and row[3] == ''):  #Following Mohit > Shiva > Shrey
					label_know = row[1].split('/')[0]
					Y_data.append(mapping_know[label_know.strip()])
				elif(row[5] == '' and row[3] != ''):
					label_know = row[3].split('/')[0]
					Y_data.append(mapping_know[label_know.strip()])
				else:
					label_know = row[5].split('/')[0]
					Y_data.append(mapping_know[label_know.strip()])

	if shuffle:
		X = list(zip(X_data, Y_data))
		random.shuffle(X)
		X_data = [x[0] for x in X]
		Y_data = [x[1] for x in X] 

	return X_data, Y_data


