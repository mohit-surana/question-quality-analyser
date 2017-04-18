import re

from nsquared import DocumentClassifier
from nsquared_v2 import predict_know_label, get_know_models

from utils import get_modified_prob_dist

subject = 'OS'

know_models = get_know_models(subject)
print('Knowledge models loaded')


def get_probabilities(question):
    level_know, prob_know = predict_know_label(question, know_models)
    array_know = get_modified_prob_dist(prob_know)

    print(question)
    print(array_know)

if __name__ == '__main__':
    chapter_knowledge_levels = {}
    with open('resources/OS[chapters]/__CourseObjectives.txt') as f:
        lines = f.read().split('\n')
        chapter_name = ''
        for line in lines:
            if '.txt)' in line:
                result = re.search('(\d+)\((\d+\.txt)\) (.*)', line)
                chapter_number = result.groups(1)
                page_number = result.groups(2)
                chapter_name = result.groups(3)
                chapter_knowledge_levels[chapter_name] = list()
            else:
                print(get_probabilities(line))
                chapter_knowledge_levels[chapter_name].append(line.strip())
    print(chapter_knowledge_levels)
