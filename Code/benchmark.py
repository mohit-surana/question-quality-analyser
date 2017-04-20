import numpy as np
import codecs
import csv
from tkinter import *

from brnn        import BiDirectionalRNN, sent_to_glove, clip
print('Done 1')
from svm_glove   import TfidfEmbeddingVectorizer
print('Done 2')
from maxent      import features
print('Done 3')
from cogvoter    import predict_cog_label, get_cog_models
print('Done 4')

from nsquared    import DocumentClassifier
print('Done 5')
from nsquared_v2 import predict_know_label, get_know_models
print('Done 6')

from utils       import get_modified_prob_dist
print('Done 7')

subject = 'ADA'
    
know_models = get_know_models(subject)
print('[Visualize] Knowledge models loaded')

cog_models = get_cog_models()
print('[Visualize] Cognitive models loaded')
questions = []
level_cogs = []
level_knows = []
idis = []
count = 0 
with open('datasets/ADA_SO_Questions.csv', 'r', encoding="latin-1") as f:
    reader = csv.reader(f.read().splitlines()[:200])
    for row in reader:
        idis.append(row[0])
        question = row[1]
        level_know, prob_know = predict_know_label(question, know_models)
        
        level_cog, prob_cog = predict_cog_label(question, cog_models)
        questions.append(question)
        level_cogs.append(level_cog)
        level_knows.append(level_know)
        count += 1
        if(count % 10 == 0):
            print(count)    

count = 0        
with codecs.open('datasets/ADA_SO_Questions_labelled.csv', 'w', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        #csvwriter.writerow(['Questions', 'Manual Label', 'NSQ', 'LDA', 'LSA', 'Knowledge', 'Cognitive'])
        for i, q, k, c in zip(idis, questions, level_knows, level_cogs):
            csvwriter.writerow([i, q, k, c])
            count += 1
            if(count % 10 == 0):
                print(count)    