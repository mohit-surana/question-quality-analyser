import sys
import os
import re

import nltk
from nltk import word_tokenize

__skip_files={'__', '.DS_Store', 'Key Terms, Review Questions, and Problems', 'Recommended Reading and Web Sites', 'Recommended Reading', 'Exercises', }


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

            file_contents = ' '.join(nltk.word_tokenize(re.sub('[^a-zA-Z ]', '', re.sub('[\s]+', ' ', re.sub('-[\s]+', '', file_contents.replace('\n', ' ').replace('--', ' ') , flags=re.M | re.DOTALL), flags=re.M | re.DOTALL), flags=re.M | re.DOTALL)))
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