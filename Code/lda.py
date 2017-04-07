from gensim import corpora, models, similarities, utils
from collections import defaultdict
import os
import csv
import re
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

LOG = True
subject = 'OS'
if(LOG):
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

import string
regex = re.compile('[%s]' % re.escape(string.punctuation))

def clean(text):
    stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r', encoding="utf8").read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
    stopwords.update(['"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    tokens = [word.lower() for word in text.split() if word.lower() not in stopwords]
    #return ' '.join([i for i in [j for j in tokens if re.match('[a-z]', j)]])
    return [i for i in [j for j in tokens if re.match('[a-z]', j)]]


doc = dict()
documents = list()
listofsections = list()
for root, dirs, files in os.walk('resources/%s' % (subject, ), topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            if '.DS_Store' not in filename and '__' not in filename:
                f = open(filename, encoding="latin-1")
                lines = ' '.join(clean(f.read()))
                doc[filename] = lines

for key, d in doc.items():
    documents.append(d)
    listofsections.append(key)


stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]    for document in documents]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]

#--------------------PREPARE MODEL -------------------------------#
def prepare_model(subject):
    print('Preparing model')
    dictionary = corpora.Dictionary(texts)
    dictionary.save('models/LDA/%s/dictionary.dict' % (subject, ))  # store the dictionary, for future reference
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('models/LDA/%s/corpus.mm' % (subject, ), corpus)  # store to disk, for later use


    id2word = corpora.Dictionary.load('models/LDA/%s/dictionary.dict' % (subject, ))
    mm = corpora.MmCorpus("models/LDA/%s/corpus.mm" % (subject, ))
    tfidf = models.TfidfModel(mm) # step 1 -- initialize a model
    tfidf.save('models/LDA/%s/tfidf_model' % (subject, ))
    corpus_tfidf = tfidf[mm]
    corpora.MmCorpus.serialize('models/LDA/%s/corpus_tfidf.mm' % (subject, ), corpus_tfidf)  # store to disk, for later use

    lda_tfidf = models.LdaModel(corpus=utils.RepeatCorpus(corpus_tfidf, 10000),
                             id2word=id2word,
                             num_topics=len(documents),
                             update_every=1,
                             chunksize=1000,
                             passes=2,
                             iterations=1000)
    print('Model Prepared for tfidf')
    lda = models.LdaModel( corpus=utils.RepeatCorpus(mm, 10000),
                        id2word=id2word,
                        update_every=1,
                        num_topics=len(documents),
                        chunksize=1000,
                        passes=2,
                        iterations=1000)
    print('Model prepared for bow')
    lda.save('models/LDA/%s/lda.model' % (subject, ))
    lda_tfidf.save('models/LDA/%s/lda_tfidf.model' % (subject, ))
    print('Prepare and save index similarities for Bow and Tfidf')

    corpus_tfidf = corpora.MmCorpus("models/LDA/%s/corpus_tfidf.mm" % (subject, ))
    index = similarities.MatrixSimilarity(lda[mm])
    index_tfidf = similarities.MatrixSimilarity(lda_tfidf[corpus_tfidf], num_features=corpus_tfidf.num_terms)
    index.save("models/LDA/%s/simIndex.index" % (subject, ))
    index_tfidf.save("models/LDA/%s/simIndex_tfidf.index" % (subject, ))

#------------------------LOAD MODEL AND TEST ------------------------------#
def load_model(subject):
    #print('Loading model')
    id2word = corpora.Dictionary.load('models/LDA/%s/dictionary.dict' % (subject, ))
    mm = corpora.MmCorpus("models/LDA/%s/corpus.mm" % (subject, ))
    corpus_tfidf = corpora.MmCorpus("models/LDA/%s/corpus_tfidf.mm" % (subject, ))
    lda = models.LdaModel.load('models/LDA/%s/lda.model' % (subject, ))
    lda_tfidf = models.LdaModel.load('models/LDA/%s/lda_tfidf.model' % (subject, ))
    index = similarities.MatrixSimilarity(lda[mm])
    index_tfidf = similarities.MatrixSimilarity(lda_tfidf[corpus_tfidf], num_features=corpus_tfidf.num_terms)
    index.load("models/LDA/%s/simIndex.index" % (subject, ))
    index_tfidf.load("models/LDA/%s/simIndex_tfidf.index" % (subject, ))
    tfidf = models.TfidfModel.load('models/LDA/%s/tfidf_model' % (subject, ))
    return id2word, lda, index, lda_tfidf, index_tfidf, tfidf

#----------------------TESTING-------------#
def run_all_questions():
    finalsims = list()
    with open('datasets/ADA_Exercise_Questions.csv', 'r', encoding="cp1252") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            sentence = ''.join(row)
            print(sentence)
            doc = ' '.join(clean(sentence))
            #print(doc)
            vec_bow = id2word.doc2bow(doc.lower().split())
            vec_lda = lda[vec_bow]
            sims = index[vec_lda]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            sims = sims[:10]
            for i in range(10):
                if(sims[i][1] != 0):
                    finalsims.append((listofsections[sims[i][0]], sims[i][1]))
            for i in finalsims:
                print(i)
            finalsims = list()
            print()
            print()

LOADED = False
def get_vector(prep_model, new_question, do_what='tfidf', subject_param = 'ADA'):
    global subject
    subject = subject_param
    
    global id2word, lda, index, lda_tfidf, index_tfidf, tfidf, LOADED
    finalsims = list()
    probs = [0.0] * 4

    doc = ' '.join(clean(new_question))
    if(prep_model == 'y'):
        prepare_model()
    if not LOADED:
        id2word, lda, index, lda_tfidf, index_tfidf, tfidf = load_model(subject_param)
        LOADED = True
    #print('Testing')
    doc = ' '.join(clean(new_question))
    if(do_what == 'bow'):
        vec_bow = id2word.doc2bow(doc.lower().split())
        vec_lda = lda[vec_bow]
        sims1 = index[vec_lda]


        sims = sorted(enumerate(sims1), key=lambda item: -item[1])
        sims = sims[:10]
        finalsims = list()
        for i in range(len(sims)):
            if(sims[i][1] != 0):
                finalsims.append((listofsections[sims[i][0]], sims[i][1]))
                #print(sims[i])
        #normalize
        finalsims_values = [s[1] for s in finalsims]
        finalsims_sections = [s[0] for s in finalsims]
        #finalsims_values = [(s - min(finalsims_values))/ (max(finalsims_values) - min(finalsims_values)) for s in finalsims_values]
        finalsims_values = [s / sum(finalsims_values) for s in finalsims_values]

        finalsims = zip(finalsims_sections, finalsims_values)
        #finalsims = [(s[0], (s[1] / len(finalsims))) for s in finalsims]
        sims1 = finalsims_values

        '''
        for i in finalsims:
            print(i)
        '''

        if len(finalsims_values) > 0:
            highest_prob = finalsims_values[0]
        else:
            highest_prob = 0
        hardcoded_matrix = [1.0, 0.6, 0.3, 0.1, 0]

        level = 0
        for i, r in enumerate(zip(hardcoded_matrix[1:], hardcoded_matrix[:-1])):
            if r[0] <= highest_prob < r[1]:
                level = i
                break

        for i in range(level):
            probs[i] = (i + 1) * highest_prob / (level * (level + 1) / 2)
        probs[level] = highest_prob
        #print('probs for bow' , probs)

    #----------------------------------------------------------------------#
    #------------------------NOW DO FOR TFIDF -----------------------------#
    if(do_what == 'tfidf'):
        vec_bow = id2word.doc2bow(doc.lower().split())
        vec_lda = lda_tfidf[vec_bow]
        sims_tfidf1 = index_tfidf[vec_lda]
        sims_tfidf = sorted(enumerate(sims_tfidf1), key=lambda item: -item[1])
        finalsims_tfidf = list()
        for i in range(len(sims_tfidf)):
            if(sims_tfidf[i][1] != 0):
                finalsims_tfidf.append((listofsections[sims_tfidf[i][0]], sims_tfidf[i][1]))
        #normalize
        finalsims_tfidf_values = [s[1] for s in finalsims_tfidf]
        finalsims_tfidf_sections = [s[0] for s in finalsims_tfidf]
        finalsims_tfidf_values = [s / sum(finalsims_tfidf_values) for s in finalsims_tfidf_values]
        finalsims_tfidf = zip(finalsims_tfidf_sections, finalsims_tfidf_values)
        sims1 = finalsims_tfidf_values

        if len(finalsims_tfidf_values) > 0:
            highest_prob = finalsims_tfidf_values[0]
        else:
            highest_prob = 0
        hardcoded_matrix = [1.0, 0.6, 0.3, 0.1, 0]

        level = 0
        for i, r in enumerate(zip(hardcoded_matrix[1:], hardcoded_matrix[:-1])):
            if r[0] <= highest_prob < r[1]:
                level = i
                break

        probs_tfidf = [0.0] * 4
        for i in range(level):
            probs_tfidf[i] = (i + 1) * highest_prob / (level * (level + 1) / 2)
            probs_tfidf[i] = (i + 1) * highest_prob / (level + 1)
        probs_tfidf[level] = highest_prob
        #print('probs for tfidf' , probs_tfidf)
        probs = probs_tfidf

    if not len(sims1):
        sims1.append(0)
    return sims1, probs

if __name__ == "__main__":
    #run_all_questions()
    questions = '''Write a very long and vague sentence that won't have too much similarity with what we have trained it with before

why is sherry's code going haywire?

apply mergesort to get a solution for the following

How would you arrange 1000 numbers such that each number is smaller than the one to its right?

DNA pattern related work is often intractable by normal methods. Design an optimized method to search for patterns

Having some problems implementing a quicksort sorting algorithm in java. I get a stackoverflow error when I run this program and I'm not exactly sure why. If anyone can point out the error, it would be great.

Apply the Horner's rule on the following

Give an example that shows that the approximation sequence of Newton's method may diverge.

Find the number of comparisons made by the sentinel version of linear search a. b. in the worst case. in the average case if the probability of a successful search is p.

Write a brute force backtracking program for playing the game Battleship on the computer.'''

    TRAIN = False
    if TRAIN:
        prepare_model(subject)

    TEST = False
    if TEST:
        for query in questions.split('\n\n'):
            print('Query:', query)
            s, p = get_vector('n', query, 'tfidf')
            print()
