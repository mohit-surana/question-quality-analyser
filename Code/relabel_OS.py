import codecs
import csv
import re

from utils import clean

import classifier as Nsq
from classifier import DocumentClassifier
import docsim_lda
import docsim_lsa

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
        
        #Following Mohit > Shiva > Shrey
        shrey_cog, shiva_cog, mohit_cog = row[2].split('/')[0].strip(), row[4].split('/')[0].strip(), row[6].split('/')[0].strip()
        label_cog = mohit_cog if mohit_cog else (shiva_cog if shiva_cog else shrey_cog)
        label_cog = label_cog.strip()
        Y_cog.append(mapping_cog[label_cog])
        
        shrey_know, shiva_know, mohit_know = row[1].split('/')[0].strip(), row[3].split('/')[0].strip(), row[5].split('/')[0].strip()
        label_know = mohit_know if mohit_know else (shiva_know if shiva_know else shrey_know)
        label_know = label_know.strip()
        Y_know.append(mapping_know[label_know])
        
train_lda = 1
with codecs.open('datasets/OS_Exercise_Questions_Relabelled.csv', 'w', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Questions', 'Manual Label', 'NSQ', 'LDA', 'LSA', 'Cognitive'])
    for x, y_cog, y_know in zip(X, Y_cog, Y_know):
        #print(x)
        #nsq = max(Nsq.get_knowledge_probs(x, 'OS'))
        nsq = 1     # TODO : SHIVA CHANGE IT
        if train_lda == 0:
            lda = max(docsim_lda.get_vector('y', x, 'tfidf', subject = 'OS')[1])
            train_lda = 1
        else:
            lda = max(docsim_lda.get_vector('n', x, 'tfidf', subject = 'OS')[1])
        lsa = docsim_lsa.get_values(x)
        csvwriter.writerow([x, y_cog + 6 * y_know, nsq, lda, lsa, y_cog])

# train_lda = 1
# with codecs.open('datasets/ADA_Exercise_Questions_Relabelled_v3.csv', 'w', encoding="utf-8") as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # NOTE: This does not have the headers and must be appropriately handled by modifying the code
#     for x, y_cog, y_know in zip(X, Y_cog, Y_know):
#         #print(x)
#         nsq = max(Nsq.get_knowledge_probs(x, 'ADA'))
#         if train_lda == 0:
#             lda = max(docsim_lda.get_vector('y', x, 'tfidf')[0])
#             train_lda = 1
#         else:
#             lda = max(docsim_lda.get_vector('n', x, 'tfidf')[0])
#         lsa = docsim_lsa.get_values(x)
#         csvwriter.writerow([x, y_cog * 16 + 100 * y_know, nsq, lda, lsa])


'''
train_lda = 1
with codecs.open('datasets/ADA_Exercise_Questions_Relabelled_v4.csv', 'w', encoding="utf-8") as csvfile:

    csvwriter = csv.writer(csvfile)
    # NOTE: This does not have the headers and must be appropriately handled by modifying the code
    for x, y_cog, y_know in zip(X, Y_cog, Y_know):
        #print(x)
        nsq = max(Nsq.get_knowledge_probs(x, 'ADA'))
        if train_lda == 0:
            lda = max(docsim_lda.get_vector('y', x, 'tfidf')[0])
            train_lda = 1
        else:
            lda = max(docsim_lda.get_vector('n', x, 'tfidf')[0])
        lsa = docsim_lsa.get_values(x)
        csvwriter.writerow([x, y_cog * 16 + 100 * y_know, nsq, lda, lsa, y_know])
'''
