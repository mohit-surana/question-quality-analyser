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

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}

if(platform.system() == 'Windows'):
    stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r', encoding='utf8').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
else:
    stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

regex = re.compile('[%s]' % re.escape(string.punctuation))
#see documentation here: http://docs.python.org/2/library/string.html

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

def clean_no_stopwords(text, as_list=True):
    tokens = [porter.stem(wordnet.lemmatize(w)) for w in text.lower().split() if w.isalpha()]
    if as_list:
        return tokens
    else:
        return ' '.join(tokens)

        
def get_glove_vector(questions):
    classify = joblib.load('models/glove_svm_model.pkl')

    print('Formatting questions')
    for i in range(len(questions)):
        questions[i] = word_tokenize(questions[i].lower())
    print('Transforming')
    print(questions)
    return classify.named_steps['word2vec vectorizer'].transform(questions, mean=False)


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

def get_data_for_cognitive_classifiers(threshold=[0, 0.1, 0.15], what_type=['ada', 'os', 'bcl'], split=0.7, include_keywords=True, keep_dup=False):
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
        random.shuffle(dataset)
        X_train = [x[0] for x in dataset]
        Y_train = [y[1] for y in dataset]

    return X_train, Y_train, X_test, Y_test
