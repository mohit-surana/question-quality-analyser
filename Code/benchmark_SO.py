############################            <deprecate>         ########################################
import csv
import pickle
import re
import platform
import codecs

import numpy as np
import nltk
from nltk.corpus import stopwords as stp
from gensim import corpora, models, similarities
from gensim.matutils import cossim, sparse2full
from nltk import stem
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.externals import joblib

from sklearn.metrics.pairwise import cosine_similarity

stemmer = stem.porter.PorterStemmer()
wordnet = WordNetLemmatizer()


subject = 'ADA'


punkt = {',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '.', '?', '!', '`', '|', '-', '=', '+', '_', '>', '<'}

if(platform.system() == 'Windows'):
    stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r', encoding='utf8').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
else:
    stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])

stopwords.update(punkt)

stopwords2 = stp.words('english')

docs, texts = None, None



def preprocess(text, stop_strength=0, remove_punct=True):
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

def load_data(subject):
    docs = {}
    cog_labels = []
    know_labels = []
    all_questions = []
    with open('datasets/%s_Exercise_Questions_Labelled.csv' % (subject, ), 'r') as f:
        reader = csv.reader(f.read().splitlines()[1:])
        for row in reader:
            title = row[0]
            temp_ques = preprocess(row[0], stop_strength=1, remove_punct=False)
            all_questions.append(temp_ques)
            cog_labels.append(row[1])
            know_labels.append(row[2])
            docs[title] = temp_ques
        
    doc_set = list(docs.values())

    texts = []
    for i in doc_set:
        texts.append(preprocess(i, stop_strength=1).split())

    return docs, texts, all_questions, know_labels, cog_labels
            
            

TRAIN_LDA = False
TEST = True
print('Loading Data')
docs, texts, all_questions, know_labels, cog_labels = load_data(subject)
print('len of docs:', len(docs))
#print('docs', docs)
#print('texts', texts)
#print('Data Loading Done')
dictionary = corpora.Dictionary(texts)
dictionary.save('models/benchmark/%s/dictionary.dict' %subject)  # store the dictionary, for future reference

corpus = []
for k in docs:
    corpus.extend([dictionary.doc2bow(sentence.split()) for sentence in nltk.sent_tokenize(docs[k])])
corpora.MmCorpus.serialize('models/benchmark/%s/corpus.mm' %subject, corpus)  # store to disk, for later use

#tfidf_model = models.TfidfModel(corpus, id2word=dictionary, normalize=True)
#tfidf_model.save('models/Nsquared/%s/tfidf.model' %subject)
        
if TRAIN_LDA:
    
    lda = models.LdaModel(corpus=corpus,
                          id2word=dictionary,
                          num_topics=len(docs),
                          update_every=1,
                          passes=2)
    # Hack to fix a big

    lda.minimum_phi_value = 0
    lda.minimum_probability = 0
    lda.save('models/benchmark/%s/lda.model' % (subject, ))
    
    print('Model training done')
else:
    lda = models.LdaModel.load('models/benchmark/%s/lda.model' % (subject, ))
    lda.minimum_phi_value = 0
    lda.minimum_probability = 0
    lda.per_word_topics = False
    print('Model Loaded')
            
if TEST:
    questions = []
    p_list = []
    know_match = []
    cog_match = []
    final_dict = {}
    s1 = []
    idis = []
    print('Testing Started')
    #Get a mapping between the questions and the document ids - final_dict
    for ques in all_questions:
                s1.append(lda[dictionary.doc2bow(ques.split())])
                #d1 = sparse2full(s1, lda.num_topics)
                #s1.sort(key = lambda x:-x[1])
                #final_dict[s1[0][0]] = ques
    print('final_dict', len(final_dict))
    count = 0
    with open('datasets/ADA_SO_Questions.csv', 'r', encoding="latin-1") as f:
        reader = csv.reader(f.read().splitlines()[:200])
        for row in reader:
            idis.append(row[0])
            q = row[1]
            questions.append(q)
            cleaned_question = preprocess(q, stop_strength=1, remove_punct=False)
            s2 = lda[dictionary.doc2bow(cleaned_question.split())]
            #d2 = sparse2full(s2, lda.num_topics)
            for s in s1:
                p_list.append(cossim(s, s2))

            #best_pos = all_questions.index(final_dict[s2[0][0]])
            best_pos = p_list.index(max(p_list))
            best_val = max(p_list)
            p_list = []
            ques = all_questions[best_pos]
            #print(ques, best_val, cleaned_question)
            know_match.append(know_labels[best_pos])
            cog_match.append(cog_labels[best_pos])
            #print(cleaned_question, know_labels[best_pos], cog_labels[best_pos])
            count += 1
            if(count % 10 == 0):
                print(count)  
    
    count = 0        
    with codecs.open('datasets/ADA_SO_Questions_labelled_ShreyMohit.csv', 'w', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        for i, q, k, c in zip(idis, questions, know_match, cog_match):
            csvwriter.writerow([i, q, k, c])
            count += 1
            if(count % 10 == 0):
                print(count)    