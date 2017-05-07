import sys
import os
import re

import nltk
from nltk.stem.wordnet import WordNetLemmatizer

wordnet = WordNetLemmatizer()

__skip_files={'__', '.DS_Store', 'Key Terms, Review Questions, and Problems', 'Recommended Reading and Web Sites', 'Recommended Reading', 'Exercises', }

def clean_no_stopwords(text, stem=True, lemmatize=True, as_list=True):
    text = re.sub('-\n', '', text).lower()
    text = re.sub('-', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('[^a-z.?! ]', '', text)
    tokens = [w for w in text.lower().split() if w.isalpha()]
    
    if lemmatize:
        tokens = [wordnet.lemmatize(w) for w in tokens]

    if stem:
        tokens = [porter.stem(w) for w in tokens]
    
    if as_list:
        return tokens
    else:
        return ' '.join(tokens)

def __get_cleaned_section_text(subject, mode):
    content = ''
    if mode == 'section':
        path = '../resources/%s/' %subject
    else:
        path = '../resources/%s[chapters]/' %subject
    for filename in sorted(os.listdir(os.path.join(os.path.dirname(__file__), path))):
        with open(path + filename, encoding='latin-1') as f:
            file_contents = f.read().lower()
            title = file_contents.split('\n')[0].strip()
            
            if len([1 for k in __skip_files if (k in title or k in filename)]):
                continue

            file_contents = clean_no_stopwords(file_contents, stem=False, lemmatize=True, as_list=False)
            content += file_contents + ' '

    return content.strip()
    
def get_cleaned_section_text(subject, mode='section'):
    if subject == 'ADA':
        return __get_cleaned_section_text(subject, mode)
    elif subject == 'OS':
        return ' '.join([__get_cleaned_section_text('OS',  mode),
                         __get_cleaned_section_text('OS2', mode), 
                         __get_cleaned_section_text('OS3', mode),
                         __get_cleaned_section_text('OS4', mode)])

if __name__ == '__main__':
    if len(sys.argv) < 2:
       subject = 'ADA'
    
    else:
        subject = sys.argv[1]

    text = get_cleaned_section_text(subject, mode='section')

    with open(subject + '.in', 'w') as f:
        f.write(text)