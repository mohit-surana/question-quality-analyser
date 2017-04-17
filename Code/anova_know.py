import os
import pickle
import platform
import re
import sys

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas
import scipy.stats as stats
import seaborn
from gensim import corpora, models, similarities
from gensim.matutils import cossim, sparse2full
from nltk import stem
from nltk.corpus import stopwords as stp
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_data_for_knowledge_classifiers, get_knowledge_probs

stemmer = stem.porter.PorterStemmer()
wordnet = WordNetLemmatizer()

if len(sys.argv) < 2:
    subject = 'ADA'
else:
    subject = sys.argv[1]


CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

knowledge_mapping = {'Metacognitive': 3, 'Procedural': 2, 'Conceptual': 1, 'Factual': 0}

def label2know(label):
    for key in knowledge_mapping:
        if(knowledge_mapping[key] == label):
            return key

skip_files={'__', '.DS_Store', 'Key Terms, Review Questions, and Problems', 'Recommended Reading and Web Sites', 'Recommended Reading', 'Summary', 'Exercises', 'Introduction'}

punkt = {',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '.', '?', '!', '`', '|', '-', '=', '+', '_', '>', '<'}

if(platform.system() == 'Windows'):
    stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r', encoding='utf8').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
else:
    stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])

stopwords.update(punkt)

stopwords2 = stp.words('english')

def __preprocess(text, stop_strength=0, remove_punct=True):
    text = re.sub('-\n', '', text).lower()
    text = re.sub('\n', ' ', text)
    if remove_punct:
        text = re.sub('[^a-z ]', '', text)
    else:
        text = re.sub('[^a-z.?! ]', '', text)
    
    tokens = [word for word in text.split()]
    if stop_strength == 0:
        return ' '.join([wordnet.lemmatize(i) for i in tokens])
    elif stop_strength == 1:
        return ' '.join([wordnet.lemmatize(i) for i in tokens if i not in stopwords2])
    else:
        return ' '.join([wordnet.lemmatize(i) for i in tokens if i not in stopwords])


docs = {}
for file_name in sorted(os.listdir('resources/%s' % (subject, ))):
    with open(os.path.join('resources', subject, file_name), encoding='latin-1') as f:
        content = f.read() #re.split('\n[\s]*Exercise', f.read())[0]
        title = content.split('\n')[0]
        if len([1 for k in skip_files if (k in title or k in file_name)]):
            continue

        docs[title] = __preprocess(content, stop_strength=2, remove_punct=False)

doc_set = list(docs.values())

texts = []
for i in doc_set:
    texts.append(__preprocess(i, stop_strength=1).split())

MODEL = ['LDA', 'LSA', 'D2V']

USE_MODELS = MODEL[0:1]

if MODEL[0] in USE_MODELS:
    dictionary = corpora.Dictionary.load('models/Nsquared/%s/dictionary.dict' % (subject, ))
    corpus = corpora.MmCorpus('models/Nsquared/%s/corpus.mm' % (subject, ))
    lda = models.LdaModel.load('models/Nsquared/%s/lda.model' % (subject, ))

if MODEL[1] in USE_MODELS:
    dictionary = corpora.Dictionary.load('models/Nsquared/%s/dictionary.dict' %subject)
    corpus = corpora.MmCorpus("models/Nsquared/%s/corpus.mm" %subject)
    tfidf_model = models.TfidfModel.load('models/Nsquared/%s/tfidf.model' %subject)
    lsi_model = models.LsiModel.load('models/Nsquared/%s/lsi.model' %subject)
    index = similarities.MatrixSimilarity(lsi_model[tfidf_model[corpus]], num_features=lsi_model.num_topics)
    index.save('models/Nsquared/%s/lsi.index' %subject)

if MODEL[2] in USE_MODELS:
    d2v_model = models.doc2vec.Doc2Vec.load('models/Nsquared/%s/d2v.model' %subject)

clf = pickle.load(open('models/Nsquared/%s/nsquared.pkl' % (subject, ), 'rb'))

x_data, y_data = get_data_for_knowledge_classifiers(subject)

print()
y_probs = []
nCorrect = nTotal = 0
for x, y in zip(x_data, y_data):
    print(CURSOR_UP_ONE + ERASE_LINE + '[N-Squared] Testing For', (nTotal + 1))
    c, k = clf.classify([x])[0]
    cleaned_question = __preprocess(x, stop_strength=1, remove_punct=False)
    
    p_list = []
    if MODEL[0] in USE_MODELS:
        s1 = lda[dictionary.doc2bow(docs[c].split())]
        s2 = lda[dictionary.doc2bow(cleaned_question.split())]
        d1 = sparse2full(s1, lda.num_topics)
        d2 = sparse2full(s2, lda.num_topics)
        lda_p = cossim(s1, s2)
        p_list.append(lda_p)

    if MODEL[1] in USE_MODELS:
        s1 = lsi_model[tfidf_model[dictionary.doc2bow(docs[c].split())]]
        s2 = lsi_model[tfidf_model[dictionary.doc2bow(cleaned_question.split() )]]
        lsa_p = cossim(s1, s2)
        p_list.append(lsa_p)

    if MODEL[2] in USE_MODELS:
        d1 = np.mean([d2v_model.infer_vector(s.split()) for s in nltk.sent_tokenize(docs[c])], axis=0).reshape(1, -1)
        d2 = d2v_model.infer_vector(cleaned_question.split()).reshape(1, -1)
        d2v_p = cosine_similarity(d1, d2)[0][0]
        p_list.append(d2v_p)
    
    p_list.append(k)
    y_pred = np.argmax(get_knowledge_probs(abs(p_list[0])))
    y_probs.append(p_list)
    if y_pred == y:
        nCorrect += 1

    nTotal += 1

y_probs = np.array(y_probs)

nsq, lda = {}, {}
for prob_vals, label in zip(y_probs, y_data):
    nsq_p, lda_p = prob_vals[0], prob_vals[1]
    if(label not in nsq):
        nsq[label], lda[label] = list(), list()
    nsq[label].append(nsq_p)
    lda[label].append(lda_p)

for label in range(3):
    print(label)
    data = pandas.DataFrame({'nsq': nsq[label], 'lda': lda[label]})
    seaborn.pairplot(data, vars=['nsq', 'lda'], kind='reg')
    # plt.show()
plt.show()

print('p values')
for label in range(3):
    print(label2know(label) + ':', stats.kruskal(nsq[label], lda[label])[1])

import matplotlib.pyplot as plt
import scipy.stats

for label in range(3):
    size = len(nsq[label])
    x = scipy.arange(size)
    y = nsq[label]
    h = plt.hist(y, color='b', normed=True, stacked=True, bins=8)
    '''
    y2 = h[0]
    x2 = h[1]
    x3 = []
    for i in range(len(x2) - 1):
        x3.append((x2[i] + x2[i+1]) / 2)
    
    y3 = scipy.interpolate.spline(x3, y2, np.linspace(0, 1, 100))
    plt.plot(np.linspace(0, 1, 100), y3)
    '''
    hmax = max(h[0])

    dist_names = ['norm']

    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        mu, var = dist.fit(y)
        print(mu, var)
        # pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
        # plt.plot(pdf_fitted, label=dist_name)
        x = np.linspace(0, 1, size)
        plt.plot(x, hmax * mlab.normpdf(x, mu, var**0.5))
        plt.xlim(0, 1)

    plt.legend(loc='upper right')
    plt.show()
