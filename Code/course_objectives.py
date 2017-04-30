import re
import numpy as np
import pprint

from nsquared import DocumentClassifier
from nsquared_v2 import predict_know_label, get_know_models

from utils import get_modified_prob_dist

subject = 'OS'

knowledge_mapping = {'Metacognitive': 3, 'Procedural': 2, 'Conceptual': 1, 'Factual': 0}
knowledge_mapping2 = { v : k for k, v in knowledge_mapping.items()}

know_models = get_know_models(subject)
print('Knowledge models loaded')


def get_probabilities(question):
    level_know, prob_know = predict_know_label(question, know_models)
    array_know = get_modified_prob_dist(prob_know)

if __name__ == '__main__':
    chapter_knowledge_levels = {}
    section_wise_objectives = {}
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
                c, k = know_models[0].classify([line.strip()])[0]
                if chapter_name not in section_wise_objectives:
                    section_wise_objectives[chapter_name] = { 'obj': { }, 'questions' : { } }

                know_level = np.argmax(get_probabilities(line.strip()))
                if c not in section_wise_objectives[chapter_number]['obj'].keys():
                    section_wise_objectives[chapter_name]['obj'][c] = set()

                section_wise_objectives[chapter_name]['obj'][c].add(know_level)
                chapter_knowledge_levels[chapter_name].append(know_level)

    pprint.pprint(section_wise_objectives)