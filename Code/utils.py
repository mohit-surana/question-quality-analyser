import re
import string

import nltk 
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.externals import joblib

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import platform
from qfilter_train import tokenizer
import dill
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

        
def get_glove_vector(questions):
    classify = joblib.load('models/glove_svm_model.pkl')

    print('Formatting questions')
    for i in range(len(questions)):
        questions[i] = word_tokenize(questions[i].lower())
    print('Transforming')
    print(questions)
    return classify.named_steps['word2vec vectorizer'].transform(questions, mean=False)        


def get_filtered_questions(questions, what_type='os'):
    questions = 'Figure 6.17 shows another solution to the dining philosophers problem using monitors. Compare to Figure 6.14 and report your conclusions.'
    what_type = 'os'

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
        
        for word in re.sub('[^a-z ]', '', questions[i].lower()).split():
            try:
                if (word_dict[word] < 0.25 or word in keywords) and word not in t_stopwords:
                    new_question += word + ' '
            except:
                pass

        new_questions.append(new_question.strip())

    return new_questions if len(new_questions) > 1 else new_questions[0]
