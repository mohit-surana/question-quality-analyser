import codecs
import csv
import re

from utils import clean

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}

X = []
Y_cog = []
Y_know = []

with codecs.open('datasets/ADA_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
    csvreader = csv.reader(csvfile.read().splitlines()[1:])
    # NOTE: This is used to skip the first line containing the headers
    for row in csvreader:
        sentence, label_cog, label_know = row
        m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
        sentence = m.groups()[2]
        label_cog = label_cog.split('/')[0]
        label_know = label_know.split('/')[0]
        # sentence = clean(sentence)
        # NOTE: Commented the above line because the cleaning mechanism is different for knowledge and cognitive dimensions
        X.append(sentence)
        Y_cog.append(mapping_cog[label_cog])
        Y_know.append(mapping_know[label_know])

with codecs.open('datasets/ADA_Exercise_Questions_Relabelled.csv', 'w', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    # NOTE: This does not have the headers and must be appropriately handled by modifying the code
    for x, y_cog, y_know in zip(X, Y_cog, Y_know):
        csvwriter.writerow([x, y_cog + 6 * y_know])