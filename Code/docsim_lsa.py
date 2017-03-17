from gensim import corpora, models, similarities, utils
from collections import defaultdict
import os
import csv
import re
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
import logging
import sys
import io
import csv
import multiprocessing
from multiprocessing import Manager
import platform

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'  

regex = re.compile('[%s]' % re.escape(string.punctuation))

subject = 'ADA' # sys.argv[1]


def run(tid, chunk, return_dict):    
    results = []  
    #print('[Process-%d]\t Starting' %tid)
    i = 0
    j = 0
    #print('\n')
    for query in chunk:
        #print('query', query)
        #if len(query.split()) < 5:
        #    j += 1
        #    continue

        query_bow = id2word.doc2bow(clean(query).lower().split())
        query_lsi = lsi_tfidf[query_bow]

        sims = index[query_lsi]
        sorted_similarities = sorted(enumerate(sims), key=lambda item: -item[1])[:10]
        sim_sum = sum([x[1] for x in sorted_similarities])
        if sim_sum == 0:
            j += 1
            continue

        sims = [(s[0], (s[1] / sim_sum)) for s in sorted_similarities]
        sims_p = [x[1] for x in sims]
        
        #if max(sims_p) > 0.5:
        i += 1
        j += 1
        #print("[Process-{}]\t Dumped {} [EVAL {}; TOTAL {}] questions".format(tid, i, j, len(chunk)))
        results.append((query, max(sims_p)))
        #else:
        #    j += 1

    return_dict[tid] = results
    #print(results)
    #print('[Process-%d]\t Finished' %tid)

def clean(text):
    text = re.sub('&.*?;', '', text)
    if(platform.system() == 'Windows'):
        stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r', encoding='utf8').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
    else:
        stopwords = set(re.split(r'[\s]', re.sub('[\W]', '', open('resources/stopwords.txt', 'r').read().lower(), re.M), flags=re.M) + [chr(i) for i in range(ord('a'), ord('z') + 1)])
    stopwords.update(['"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

    tokens = [word.lower() for word in text.split() if word.lower() not in stopwords]
    return ' '.join([i for i in [j for j in tokens if re.match('[a-z]', j)]])

def cosToProb(cos, gamma=1): 
    cos_min = min(cos) 
    cos_shifted = (cos - cos_min) ** gamma 
    sum_cos_shifted = sum(cos_shifted) 
    return [float(c) / float(sum_cos_shifted) for c in cos_shifted]


TRAIN = False

doc = dict()
documents = list()
listofsections = list()
for root, dirs, files in os.walk('resources/%s' %subject, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            if re.match('[\d]', name):
                if '__' in name:
                    continue
                f = io.open(filename, "r", encoding='windows-1252')
                lines = clean(f.read())
                doc[filename] = lines

for key, d in doc.items():
    documents.append(d)
    listofsections.append(key)

if not os.path.exists('models/lsa/%s_dictionary.dict' %subject):
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]    for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    dictionary = corpora.Dictionary(texts)
    dictionary.save('models/lsa/%s_dictionary.dict' %subject)  # store the dictionary, for future reference
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('models/lsa/%s_corpus.mm' %subject, corpus)  # store to disk, for later use

id2word = corpora.Dictionary.load('models/lsa/%s_dictionary.dict' %subject)
mm = corpora.MmCorpus("models/lsa/%s_corpus.mm" %subject)

tfidf = models.TfidfModel(mm, normalize=True)

# creates a lazy evaluating wrapper around corpus 
corpus_tfidf = tfidf[mm]

if TRAIN:
    lsi_tfidf = models.LsiModel(utils.RepeatCorpus(corpus_tfidf, 10000), 
                                id2word=id2word, 
                                onepass=False,
                                extra_samples=300,
                                num_topics=len(documents))
    lsi_tfidf.save('models/lsa/%s_lsi.model')
    print('Done training')

lsi_tfidf = models.LsiModel.load('models/lsa/%s_lsi.model' %subject)
corpus_lsi_tfidf = lsi_tfidf[corpus_tfidf]
index = similarities.MatrixSimilarity(corpus_lsi_tfidf, num_features=lsi_tfidf.num_topics)
index.save('models/lsa/%s_lsi.index' %subject)

'''
queries = ["How would you arrange 1000 numbers such that each number is smaller than the one to its right?",

"DNA pattern related work is often intractable by normal methods. Design an optimized method to search for patterns",

"Having some problems implementing a quicksort sorting algorithm in java. I get a stackoverflow error when I run this program and I'm not exactly sure why. If anyone can point out the error, it would be great.",

"Give an example that shows that the approximation sequence of Newton's method may diverge.",

"Find the number of comparisons made by the sentinel version of linear search a. b. in the worst case. in the average case if the probability of a successful search is p (0 p 1).",

"Write a brute force pattern matching program for playing the game Battleship on the computer."]
'''
def get_values(question):
    return_dict = dict()
    run(1,[question],return_dict)
    #print(return_dict)
    for v in return_dict.values():
        #print(v)
        if(len(v) == 0):
            return 0
        return v[0][1]

if __name__ == '__main__':
    queries = []
    with open('datasets/%s_Exercise_Questions_Relabelled_v2.csv' %subject) as f:
        csvreader = csv.reader(f)
        i = 0
        for row in csvreader:
            queries.append(row[0])
            i += 1

    manager = Manager()
    return_dict = manager.dict()
    #queries = queries[:100]
    #print(queries)
    num_procs = 8
    procs = []
    for chunk in [queries[i::num_procs] for i in range(num_procs)]:
        p = multiprocessing.Process(target=run, args=(len(procs), chunk, return_dict))
        procs.append(p)
        p.start()

    results = []
    for proc in procs:
        proc.join()
    #print(return_dict)
    for v in return_dict.values():
        results.extend(v)
    #print(results)

    with open('datasets/LSA_Questions_Labelled.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(results)
            
    #print("\n".join(["%.3f - %s" % (score, listofsections[i]) for i, score in sorted_similarities[:5]]))
    #print('-' * 80)
    

