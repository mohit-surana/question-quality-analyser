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
import pickle

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

def get_filtered_questions(questions, what_type='os'):
    from qfilter_train import f
    
    t_stopwords = set(nltk.corpus.stopwords.words('english'))

    try:
        domain = pickle.load(open('resources/domain.pkl',  'rb'))
    except:
        domain = pickle.load(open('resources/domain_2.pkl',  'rb'))

    keywords = set()
    for k in domain:
        keywords = keywords.union(set(list(map(str.lower, map(str, list(domain[k]))))))
    t_stopwords = t_stopwords - keywords

    if type(questions) != list:
        questions = [questions]
    sklearn_tfidf = pickle.load(open('models/tfidf_filterer_%s.pkl' %what_type.lower(), 'rb'))
    tfidf_matrix = sklearn_tfidf.transform(questions)
    feature_names = sklearn_tfidf.get_feature_names()

    new_questions = []

    for i in range(0, len(questions)):
        feature_index = tfidf_matrix[i,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        word_dict = {w : s for w, s in [(feature_names[j], s) for (j, s) in tfidf_scores]}

        new_question = ''
        question = re.split('([.!?])', questions[i])
        question2 = []
        for k in range(len(question) - 1):
            if re.search('[!.?]', question[k + 1]):
                question2.append(question[k] + question[k + 1])
            elif re.search('[!.?]', question[k]):
                continue
            elif question[k] != '':
                question2.append(question[k])

        question = question2

        if len(question) >= 3:
            question2 = question[0] + question[-2]
            question2 += question[-1] if 'hint' not in question[-1] else ''
            questions[i] = question2

        for word in questions[i].lower().split():
            try:
                if (word_dict[word] < 1 or word in keywords) and word not in t_stopwords:
                    new_question += word + ' '
            except:
                pass

        new_questions.append(new_question)

    return new_questions if len(new_questions) > 1 else new_questions[0]
    