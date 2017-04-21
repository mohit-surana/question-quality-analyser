import os
os.chdir('../..')
print(os.getcwd())

import numpy as np

from cogvoter import predict_cog_label, get_cog_models
from nsquared import DocumentClassifier
from nsquared_v2 import predict_know_label, get_know_models
from utils import get_modified_prob_dist

subject = 'ADA'

know_models = get_know_models(subject)
print('[Visualize] Knowledge models loaded')
cog_models = get_cog_models()
print('[Visualize] Cognitive models loaded')


def get_probabilities(question):
    question = question.get()

    level_know, prob_know = predict_know_label(question, self.know_models)
    array_know = get_modified_prob_dist(prob_know)

    level_cog, prob_cog = predict_cog_label(question, self.cog_models, subject)
    array_cog = get_modified_prob_dist(prob_cog)

    nmarray = np.dot(np.array(array_know).reshape(-1, 1), np.array(array_cog).reshape(1, -1))

    return nmarray

os.chdir('../..')