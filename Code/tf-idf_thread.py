import threading
import time
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
import math
import glob

exitFlag = 0
def freq(word, doc):
    return doc.count(word)
    #TRY REGEX

def found(word, doc):
    return word in doc

def word_count(doc):
    return len(doc)


def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))


def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if found(word, document):
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))


def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))
    
class myThread (threading.Thread):
    def __init__(self, threadID, name, documents, docs, tok_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.documents = documents
        self.docs = docs
        self.tok_name = tok_name
    

    def run(self):
        print("Starting " + self.name)
        self.docs = calc_tf_idf(self.documents, self.tok_name)
        print("Exiting " + self.name)

def calc_tf_idf(documents, tok_name):
    vocabulary = []
    vocab = []
    docs = {key: None for key in documents}

    for document in documents:
        tokens = tokenizer.tokenize(str(documents[document]))
        final_tokens = []
        tokens = [token.lower() for token in tokens if len(token) > 2]
        if tok_name == 'unigram':
            final_tokens.extend(tokens)
        elif tok_name == 'bigram':
            bi_tokens = bigrams(tokens)
            bi_tokens = [' '.join(token).lower() for token in bi_tokens]
            final_tokens.extend(bi_tokens)

        else:
            tri_tokens = trigrams(tokens)
            tri_tokens = [' '.join(token).lower() for token in tri_tokens]
            final_tokens.extend(tri_tokens)

        docs[document] = {'freq': {}, 'tf': {}, 'idf': {},
            'tf-idf': {}, 'tokens': []}

        for token in final_tokens:
            #The frequency computed for each document
            docs[document]['freq'][token] = freq(token, final_tokens)
            #The term-frequency (Normalized Frequency)
            docs[document]['tf'][token] = tf(token, final_tokens)
            docs[document]['tokens'] = final_tokens

        vocabulary.extend(final_tokens)
        vocab.append(' '.join(final_tokens))
    
    open('resources/vocabulary_' + tok_name+'.txt', 'w').write(' ;'.join(vocabulary))

    print('calculating idf for ', tok_name)
    for doc in docs:
        for token in docs[doc]['tf']:
            #The Inverse-Document-Frequency
            docs[doc]['idf'][token] = idf(token, vocab)
            #The tf-idf
            docs[doc]['tf-idf'][token] = docs[doc]['tf'][token] * docs[doc]['idf'][token]#tf_idf(token, docs[doc]['tokens'], vocabulary)
    return docs

tokenizer = RegexpTokenizer("[\w’]+", flags=re.UNICODE)
def main():
    start_time = time.time()
    fhandler_new = open('resources/keywords_enhanced.txt', 'w')

    fhandler = open('resources/keywords.txt', 'w')
    fhandler_uni = open('resources/keywords_uni.txt', 'w')
    fhandler_bi = open('resources/keywords_bi.txt', 'w')
    fhandler_tri = open('resources/keywords_tri.txt', 'w')
    
    documents = dict()
    l =  glob.glob(r"resources\ADA-chapter\*.txt")

    for i in l:
        f = open(i, 'rb')
        documents[i] = str(f.read())

    docs = {key: None for key in documents}

    # Create new threads
    thread1 = myThread(1, "Thread-1", documents, docs, 'unigram')
    thread2 = myThread(2, "Thread-2", documents, docs, 'bigram')
    thread3 = myThread(3, "Thread-3", documents, docs, 'trigram')

    # Start new Threads
    thread1.start()
    thread2.start()
    thread3.start()
    # Add threads to thread list
    threads = []
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    print('All threads done... Starting to save in files')
    #Now let's find out the most relevant words by tf-idf for TRIGRAMS
    words = {}
    list_of_trigrams = []
    for doc in thread3.docs:
        for token in thread3.docs[doc]['tf-idf']:
            if token not in words:
                words[token] = thread3.docs[doc]['tf-idf'][token]
            else:
                if thread3.docs[doc]['tf-idf'][token] > words[token]:
                    words[token] = thread3.docs[doc]['tf-idf'][token]

        #print('Document is:', doc)
        
    for item in sorted(words.items(), key=lambda x: x[1], reverse=True):
        list_of_trigrams.append(item[0])
    #Get the five thousand best trigrams	
    #list_of_trigrams = list_of_trigrams[:5000]
    fhandler_tri.write(' ;'.join(list_of_trigrams))
    
    print('Starting to save bigrams in files')
    #Now let's find out the most relevant words by tf-idf for BIGRAMS
    words = {}
    list_of_bigrams = []
    for doc in thread2.docs:
        for token in thread2.docs[doc]['tf-idf']:
            if token not in words:
                words[token] = thread2.docs[doc]['tf-idf'][token]
            else:
                if thread2.docs[doc]['tf-idf'][token] > words[token]:
                    words[token] = thread2.docs[doc]['tf-idf'][token]
        
    for item in sorted(words.items(), key=lambda x: x[1], reverse=True):
        #if item[0] not in ' '.join(list_of_trigrams):
        list_of_bigrams.append(item[0])
    #Get the five thousand best bigrams	
    #list_of_bigrams = list_of_bigrams[:5000]
    fhandler_bi.write(' ;'.join(list_of_bigrams))
    
    print('Starting to save unigrams in files')
    #Now let's find out the most relevant words by tf-idf for UNIGRAMS
    words= {}
    list_of_unigrams = []
    for doc in thread1.docs:
        for token in thread1.docs[doc]['tf-idf']:
            if token not in words:
                words[token] = thread1.docs[doc]['tf-idf'][token]
            else:
                if thread1.docs[doc]['tf-idf'][token] > words[token]:
                    words[token] = thread1.docs[doc]['tf-idf'][token]
        
    for item in sorted(words.items(), key=lambda x: x[1], reverse=True):
        #if item[0] not in ' '.join(list_of_trigrams) and item[0] not in ' '.join(list_of_bigrams):
        list_of_unigrams.append(item[0])
    #Get the five thousand best bigrams	
    #list_of_unigrams = list_of_unigrams[:5000]
    fhandler_uni.write(' ;'.join(list_of_unigrams))
    
    list_of_words = []
    list_of_words1 = []
    list_of_bigrams1 = []
    list_of_unigrams1 = []
    list_of_trigrams1 = list_of_trigrams[:5000]
    list_of_words.extend(list_of_unigrams)
    list_of_words.extend(list_of_bigrams)
    list_of_words.extend(list_of_trigrams)		
    
    for token in list_of_words:
        fhandler.write(token+'\n')
    

    tri = ' '.join(list_of_trigrams1)

    for token in list_of_bigrams:
        if(token not in tri):
            list_of_bigrams1.append(token)
    list_of_bigrams1 = list_of_bigrams1[:5000]
    
    bi = ' '.join(list_of_bigrams1)
    for token in list_of_unigrams:
        if (token not in tri and (token not in bi )):
            list_of_unigrams1.append(token)
    list_of_unigrams1 = list_of_unigrams1[:5000]
    
    list_of_words1.extend(list_of_unigrams1)
    list_of_words1.extend(list_of_bigrams1)
    list_of_words1.extend(list_of_trigrams1)		
    
    for token in list_of_words1:
        fhandler_new.write(token+'\n')

    fhandler_new.close()

    fhandler.close()
    fhandler_tri.close()
    fhandler_bi.close()
    fhandler_uni.close()
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Exiting Main Thread")


main()
