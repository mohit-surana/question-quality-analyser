import re
import string

import nltk 
from nltk import word_tokenize
from nltk.corpus import stopwords
#if not word in stopwords.words('english'): # Loss of crucial words

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import platform
porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()


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