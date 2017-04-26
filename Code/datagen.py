import csv
import os
import random

random.seed(63905)

cog2label = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
label2cog = { v : k for k, v in cog2label.items()}

know2label = {'Metacognitive': 3, 'Procedural': 2, 'Conceptual': 1, 'Factual': 0}
label2know = { v : k for k, v in know2label.items()}

cp_know = dict()
cp_know['ADA'] = [(0, 0.2), (1, 0.5), (2, 0.95), (3, 1.0)]
cp_know['OS'] = [(0, 0.3), (1, 0.65), (2, 0.95), (3, 1.0)]

cp_cog = dict()
cp_cog['ADA'] = [(0, 0.1), (1, 0.3), (2, 0.55), (3, 0.65), (4, 0.80), (5, 1.0)]
cp_cog['OS'] = [(0, 0.25), (1, 0.5), (2, 0.60), (3, 0.80), (4, 0.90), (5, 1.0)]

difficulty_know = [0.2, 0.3, 0.4, 0.9]
difficulty_cog = [0.1, 0.2, 0.25, 0.4, 0.6, 0.75]

NO_OF_QUESTIONS = 20
NO_OF_STUDENTS = 100


def get_label(table, prob):
    for label, limit in table:
        if prob <= limit:
            return label


def create_paper(subject):
    this_cp_know = cp_know[subject]
    this_cp_cog = cp_cog[subject]
    rows = []
    for question in range(NO_OF_QUESTIONS):
        row = list()
        row.append('Q' + str(question + 1))
        r1, r2 = random.random(), random.random()
        know_label = get_label(this_cp_know, r1)
        cog_label = get_label(this_cp_cog, r2)
        row.append(know_label)
        row.append(cog_label)
        for student in range(NO_OF_STUDENTS):
            difficulty = (difficulty_know[know_label] + difficulty_cog[cog_label]) / 2
            r = random.random()
            row.append('Correct' if r > difficulty else 'Incorrect')
        rows.append(row)
    return rows


def save2csv(subject, rows):
    with open(os.path.join(os.path.dirname(__file__), 'resources', subject + '_Performance.csv'), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in rows:
            csvwriter.writerow(row)


if __name__ == '__main__':
    save2csv('ADA', create_paper('ADA'))
    save2csv('OS', create_paper('OS'))