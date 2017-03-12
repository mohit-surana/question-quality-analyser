import re
import string

from nltk import word_tokenize
from nltk.corpus import stopwords
#if not word in stopwords.words('english'): # Loss of crucial words

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

regex = re.compile('[%s]' % re.escape(string.punctuation))
#see documentation here: http://docs.python.org/2/library/string.html

def clean(sentence):
    sentence = sentence.lower()

    final_sentence = []
    for word in word_tokenize(sentence):
        word = regex.sub(u'', word)
        if not (word == u'' or word == ''):
            word = wordnet.lemmatize(word)
            word = porter.stem(word)
            #word = snowball.stem(word)
            final_sentence.append(word)
    return final_sentence
