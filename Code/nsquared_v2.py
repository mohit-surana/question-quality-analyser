import logging
import os
import dill
import pickle
import platform
import re
import sys

import nltk
import numpy as np
import gensim
from gensim import corpora, models, similarities
from gensim.matutils import cossim, sparse2full
from nltk import stem
from nltk.corpus import stopwords as stp
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import get_data_for_knowledge_classifiers, get_glove_vectors
from nsquared import DocumentClassifier
from collections import defaultdict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

stemmer = stem.porter.PorterStemmer()
wordnet = WordNetLemmatizer()

subject = 'ADA'

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

knowledge_mapping = {'Metacognitive': 3, 'Procedural': 2, 'Conceptual': 1, 'Factual': 0}

skip_files={'__', '.DS_Store', 'Key Terms, Review Questions, and Problems', 'Recommended Reading and Web Sites', 'Recommended Reading', 'Summary', 'Exercises', 'Introduction'}

punkt = {',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '.', '?', '!', '`', '|', '-', '=', '+', '_', '>', '<'}

if(platform.system() == 'Windows'):
    stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open(os.path.join(os.path.dirname(__file__), 'resources/stopwords.txt'), 'r', encoding='utf8').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
else:
    stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open(os.path.join(os.path.dirname(__file__), 'resources/stopwords.txt'), 'r').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])

stopwords.update(punkt)

stopwords2 = stp.words('english')


class TfidfEmbeddingVectorizerKnowledgeDimension(BaseEstimator, ClassifierMixin):
    def __init__(self, w2v, tfidf, subject):
        self.word2weight = None
        self.w2v = w2v
        self.dim = len(w2v['the'])
        self.subject = subject

        max_idf = max(tfidf.idf_)
        self.max_idf = max_idf

        self.word2weight = defaultdict(lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    def fit(X, y):
        return self
    
    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] * (self.word2weight[w])
                     for w in words if w in self.w2v] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


        return np.array(irritating_shit)

    def save(self):
        self.w2v = None
        joblib.dump(self, os.path.join(os.path.dirname(__file__), 'models/Nsquared/%s/glove_model.pkl' %self.subject))

    def load(subject, model_name='glove_model.pkl', w2v=None):
        if w2v == None:
            raise Exception('No w2v model specified')

        clf = joblib.load(os.path.join(os.path.dirname(__file__), 'models/Nsquared/%s/' %subject + model_name))
        clf.w2v = w2v

        return clf
    

def __preprocess(text, stop_strength=1, remove_punct=True):
    text = re.sub('-\n', '', text).lower()
    text = re.sub('-', ' ', text)
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

####################### ONE TIME MODEL LOADING #########################

def get_know_models(__subject):
    global subject
    subject = __subject
    load_texts(subject)

    nsq = pickle.load(open(os.path.join(os.path.dirname(__file__), 'models/Nsquared/%s/nsquared.pkl' % (subject, )), 'rb'))
    # for LDA
    model = models.LdaModel.load(os.path.join(os.path.dirname(__file__), 'models/Nsquared/%s/lda.model' % (subject, )))
    # for GloVe
    # w2v = load_glove(__subject, vec_size=300)
    # model = TfidfEmbeddingVectorizerKnowledgeDimension.load(__subject, w2v=w2v)
    ann = joblib.load(os.path.join(os.path.dirname(__file__), 'models/Nsquared/%s/know_ann_clf.pkl' %subject))

    dictionary = corpora.Dictionary.load(os.path.join(os.path.dirname(__file__), 'models/Nsquared/%s/dictionary.dict' % (subject, )))
    corpus = corpora.MmCorpus(os.path.join(os.path.dirname(__file__), 'models/Nsquared/%s/corpus.mm' % (subject, )))

    print('Loaded models for visualization')

    return nsq, model, ann, dictionary, corpus
    
##################### PREDICTION WITH PARAMS ############################

def predict_know_label(question, models, subject='ADA'):
    __docs, _ = load_texts(subject)

    nsq, model, ann, dictionary, corpus = models
    x = question
    p_list = []
    c, k = nsq.classify([x])[0]
    cleaned_question = __preprocess(x, stop_strength=1, remove_punct=False)

    # using GloVe model
    '''
    s1 = model.transform([docs[c].split()])[0].reshape(1, -1)
    s2 = model.transform([cleaned_question.split()])[0].reshape(1, -1)
    glove_p = cosine_similarity(s1, s2)[0]
    #p_list.append(glove_p)
    p_list.extend(list(s1[0]))
    p_list.extend(list(s2[0]))
    '''

    # using LDA model 
    s1 = model[dictionary.doc2bow(__docs[c].split())]
    s2 = model[dictionary.doc2bow(cleaned_question.split())]
    d1 = sparse2full(s1, model.num_topics)
    d2 = sparse2full(s2, model.num_topics)
    lda_p = cossim(s1, s2)
    p_list.append(lda_p)
    p_list.extend(list(d1))
    p_list.extend(list(d2))
    
    p_list.extend([k])

    return ann.predict([p_list])[0], ann.predict_proba([p_list])[0]

    
def load_texts(subject, stop_strength=1, remove_punct=False, split=True):
    global docs, texts
    docs = {}
    for file_name in sorted(os.listdir(os.path.join(os.path.dirname(__file__), 'resources/%s' % (subject, )))):
        with open(os.path.join(os.path.dirname(__file__), 'resources', subject, file_name), encoding='latin-1') as f:
            content = f.read()
            title = content.split('\n')[0]
            if len([1 for k in skip_files if (k in title or k in file_name)]):
                continue
            docs[title] = __preprocess(content, stop_strength=stop_strength, remove_punct=remove_punct)

    doc_set = list(docs.values())

    texts = []
    for i in doc_set:
        if split:
            texts.append(i.split())
        else:
            texts.append(i)
    return docs, texts

def load_glove(subject, vec_size):
    ############# GLOVE LOADING CODE ####################
    savepath = 'glove.%dd.pkl' %vec_size # use custom model by default

    w2v = {}
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath)):
        print()
        w2v.update(get_glove_vectors('resources/GloVe/' + 'glove.6B.%dd.txt' %vec_size))

        pickle.dump(w2v, open(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath), 'wb'))
    else:
        w2v = pickle.load(open(os.path.join(os.path.dirname(__file__), 'resources/GloVe/' + savepath), 'rb'))

    print()
    w2v.update(get_glove_vectors('resources/GloVe/' + 'glove.%s.%dd.txt' %(subject, vec_size))) #loading custom subject vector

    return w2v

#########################################################################
#                            MAIN BEGINS HERE                           #
#########################################################################
if __name__ == '__main__':
    MODEL = ['LDA', 'GLOVE', 'LSA', 'D2V']

    docs, texts = load_texts(subject)

    USE_MODELS = MODEL[0:1]

    dictionary = corpora.Dictionary(texts)
    dictionary.save('models/Nsquared/%s/dictionary.dict' %subject)  # store the dictionary, for future reference

    corpus = []
    for k in docs:
        corpus.extend([dictionary.doc2bow(sentence.split()) for sentence in nltk.sent_tokenize(docs[k])])
    corpora.MmCorpus.serialize('models/Nsquared/%s/corpus.mm' %subject, corpus)  # store to disk, for later use

    tfidf_model = models.TfidfModel(corpus, id2word=dictionary, normalize=True)
    tfidf_model.save('models/Nsquared/%s/tfidf.model' %subject)

    if MODEL[0] in USE_MODELS:
        TRAIN_LDA = False
        
        if TRAIN_LDA:
            lda = models.LdaModel(corpus=gensim.utils.RepeatCorpus(corpus, 10000),
                                  id2word=dictionary,
                                  num_topics=len(docs),
                                  update_every=1,
                                  passes=2)

            lda.save('models/Nsquared/%s/lda.model' % (subject, ))
            
            print('Model training done')
        else:
            lda = models.LdaModel.load('models/Nsquared/%s/lda.model' % (subject, ))
            
    if MODEL[1] in USE_MODELS:
        TRAIN_GLOVE = False

        tfidf_model_glove = pickle.load(open('models/tfidf_filterer_%s.pkl' %subject.lower(), 'rb'))
        print('Loaded Tfidf model')

        w2v = load_glove(subject, vec_size=300)
        print('Loaded Glove w2v')
        
        if TRAIN_GLOVE:
            clf_glove = TfidfEmbeddingVectorizerKnowledgeDimension(w2v, tfidf_model_glove, subject)
            clf_glove.save()
            clf_glove.w2v = w2v

            print('Saving done')
        else:
            clf_glove = TfidfEmbeddingVectorizerKnowledgeDimension.load(subject, w2v=w2v)
        
    if MODEL[2] in USE_MODELS:
        TRAIN_LSA = False
        
        if TRAIN_LSA:
            lsi_model = models.LsiModel(corpus=tfidf_model[corpus],
                                        id2word=dictionary,
                                        num_topics=len(docs),
                                        onepass=False,
                                        power_iters=2,
                                        extra_samples=300)
            lsi_model.save('models/Nsquared/%s/lsi.model' %subject)
            
            print('Model training done')
        else:
            lsi_model = models.LsiModel.load('models/Nsquared/%s/lsi.model' %subject)
        
        index = similarities.MatrixSimilarity(lsi_model[tfidf_model[corpus]], num_features=lsi_model.num_topics)
        index.save('models/Nsquared/%s/lsi.index' %subject)

    
    if MODEL[3] in USE_MODELS:
        
        TRAIN_D2V = False
        
        if TRAIN_D2V:
            x_train = []
            for i, k in enumerate(docs):
                x_train.append(models.doc2vec.LabeledSentence(docs[k].split(), [k]))
            
            d2v_model = models.doc2vec.Doc2Vec(size=128, alpha=0.025, min_alpha=0.025, window=2, min_count=2, dbow_words=1, workers=4)  # use fixed learning rate
            d2v_model.build_vocab(x_train)
            for epoch in range(15):
                d2v_model.train(x_train, total_examples=d2v_model.corpus_count, epochs=5)
                d2v_model.alpha -= 0.001
                d2v_model.min_alpha = d2v_model.alpha
            
            d2v_model.save('models/Nsquared/%s/d2v.model' %subject)
        else:
            d2v_model = models.doc2vec.Doc2Vec.load('models/Nsquared/%s/d2v.model' %subject)
    
    
    clf = pickle.load(open('models/Nsquared/%s/nsquared.pkl' % (subject, ), 'rb'))

    x_data, y_data = get_data_for_knowledge_classifiers(subject)

    '''
    ################ EQUAL DIVISION ##################
    y = np.bincount(y_data)
    ii = np.nonzero(y)[0]
    print(list(zip(ii, y[ii])))
    data_dict = { i : [] for i in range(4) }
    for x, y in zip(x_data, y_data):
        data_dict[y] = x
    x_data = []
    y_data = []
    for k in data_dict:
        x = data_dict[k][:128]
        x_data.extend(x)
        y_data.extend(list(np.repeat(k, len(x))))
    x_all_data = list(zip(x_data, y_data))
    random.shuffle(x_all_data)
    x_data = [x[0] for x in x_all_data]
    y_data = [x[1] for x in x_all_data]
    '''

    print()
    y_probs = []

    nTotal = 0
    for x, y in zip(x_data, y_data):
        print(CURSOR_UP_ONE + ERASE_LINE + '[N-Squared] Testing For', (nTotal + 1))
        c, k = clf.classify([x])[0]
        cleaned_question = x
        
        p_list = []
        if MODEL[0] in USE_MODELS:
            ################ BEGIN TESTING FOR LDA ##################
            s1 = lda[dictionary.doc2bow(docs[c].split())]
            s2 = lda[dictionary.doc2bow(cleaned_question.split())]
            d1 = sparse2full(s1, lda.num_topics)
            d2 = sparse2full(s2, lda.num_topics)
            lda_p = cossim(s1, s2)
            #p_list.append(lda_p)
            p_list.extend(list(d1))
            p_list.extend(list(d2))
        
        if MODEL[1] in USE_MODELS:
            ################ BEGIN TESTING for GLOVE ################
            s1 = clf_glove.transform([docs[c].split()])[0].reshape(1, -1)
            s2 = clf_glove.transform([cleaned_question.split()])[0].reshape(1, -1)
            glove_p = cosine_similarity(s1, s2)[0]
            #p_list.append(glove_p)
            p_list.extend(list(s1[0]))
            p_list.extend(list(s2[0]))

        if MODEL[2] in USE_MODELS:
            s1 = lsi_model[tfidf_model[dictionary.doc2bow(docs[c].split())]]
            s2 = lsi_model[tfidf_model[dictionary.doc2bow(cleaned_question.split() )]]
            d1 = sparse2full(s1, lsi_model.num_topics)
            d2 = sparse2full(s2, lsi_model.num_topics)
            lsa_p = cossim(s1, s2)
            #p_list.append(lsa_p)
            p_list.extend(list(d1))
            p_list.extend(list(d2))
        
        if MODEL[3] in USE_MODELS:
            d1 = np.mean([d2v_model.infer_vector(s.split()) for s in nltk.sent_tokenize(docs[c])], axis=0).reshape(1, -1)
            d2 = d2v_model.infer_vector(cleaned_question.split()).reshape(1, -1)
            d2v_p = cosine_similarity(d1, d2)[0][0]
            #p_list.append(d2v_p)
            p_list.extend(list(d1[0]))
            p_list.extend(list(d2[0]))
                    
        p_list.extend([k])
        y_probs.append(p_list)

        nTotal += 1
        
    x_train, x_test, y_train, y_test = train_test_split(y_probs, y_data, test_size=0.20)

    TRAIN_ANN = True
    
    ###### NEURAL NETWORK BASED SIMVAL -> KNOW MAPPING ########
    if TRAIN_ANN:
        if subject == 'OS':
            ann_clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(32, 16), batch_size=4, learning_rate='adaptive', learning_rate_init=0.001, verbose=True)
        else:
            ann_clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(32, 16), batch_size=4,
                            learning_rate='adaptive', learning_rate_init=0.001, verbose=True, max_iter=500)
        ann_clf.fit(x_train, y_train)
        print('ANN training completed')
        joblib.dump(ann_clf, 'models/Nsquared/%s/know_ann_clf.pkl' %subject)

    else:
        ann_clf = joblib.load('models/Nsquared/%s/know_ann_clf.pkl' %subject)
        
    y_real, y_pred = np.array(y_test), ann_clf.predict(x_test)

    print(classification_report(y_real, y_pred))
    print(confusion_matrix(y_real, y_pred))

    print('Accuracy: {:.2f}%'.format(accuracy_score(y_real, y_pred) * 100))
