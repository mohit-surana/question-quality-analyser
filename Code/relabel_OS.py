import codecs
import csv
import re

from utils import clean

import nsquared as Nsq
from nsquared import DocumentClassifier
import lda
import lsa

mapping_cog = {'Remember': 0, 'Understand': 1, 'Apply': 2, 'Analyse': 3, 'Evaluate': 4, 'Create': 5}
mapping_know = {'Factual': 0, 'Conceptual': 1, 'Procedural': 2, 'Metacognitive': 3}

X = []
Y_cog = []
Y_know = []

with codecs.open('datasets/OS_Exercise_Questions_Labelled.csv', 'r', encoding="utf-8") as csvfile:
    csvreader = csv.reader(csvfile.read().splitlines()[5:])
    # NOTE: This is used to skip the first line containing the headers
    for row in csvreader:
        #sentence, label_cog, label_know = row
        #m = re.match('(\d+\. )?([a-z]\. )?(.*)', sentence)
        #sentence = m.groups()[2]
        #label_cog = label_cog.split('/')[0]
        #label_know = label_know.split('/')[0]
        # sentence = clean(sentence)
        # NOTE: Commented the above line because the cleaning mechanism is different for knowledge and cognitive dimensions
        
        X.append(row[0])
        if(row[6] == '' and row[4] == ''):      #Following Mohit > Shiva > Shrey
            label_cog = row[2].split('/')[0]
            Y_cog.append(mapping_cog[label_cog.strip()])
        elif(row[6] == '' and row[4] != ''):
            label_cog = row[4].split('/')[0]
            Y_cog.append(mapping_cog[label_cog.strip()])
        else:
            label_cog = row[6].split('/')[0]
            Y_cog.append(mapping_cog[label_cog.strip()])
        
        if(row[5] == '' and row[3] == ''):
            label_know = row[1].split('/')[0]
            Y_know.append(mapping_know[label_know.strip()])
        elif(row[5] == '' and row[3] != ''):
            label_know = row[3].split('/')[0]
            Y_know.append(mapping_know[label_know.strip()])
        else:
            label_know = row[5].split('/')[0]
            Y_know.append(mapping_know[label_know.strip()])
        
count = 0        

with codecs.open('datasets/OS_Exercise_Questions_Relabelled.csv', 'w', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Questions', 'Manual Label', 'NSQ', 'LDA', 'LSA', 'Cognitive'])
    for x, y_cog, y_know in zip(X, Y_cog, Y_know):
        #print(x)
        nsq = max(Nsq.get_knowledge_probs(x, 'OS'))
        lda_label = max(lda.get_vector('n', x, 'tfidf', subject_param = 'OS')[1])
        lsa_label = lsa.get_values(x, subject_param = 'OS')
        csvwriter.writerow([x, y_cog + 6 * y_know, nsq, lda_label, lsa_label, y_cog])
        count += 1
        if(count % 10 == 0):
            print(count)