import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from cogvoter import predict_cog_label, get_cog_models
from nsquared import DocumentClassifier
from nsquared_v2 import predict_know_label, get_know_models
from utils import get_modified_prob_dist

know_models = {'ADA': get_know_models('ADA'), 'OS': get_know_models('OS')}
print('[Visualize] Knowledge models loaded')
cog_models = get_cog_models()
print('[Visualize] Cognitive models loaded')


def get_probabilities(question, subject):
    level_know, prob_know = predict_know_label(question, know_models[subject])
    array_know = get_modified_prob_dist(prob_know)

    level_cog, prob_cog = predict_cog_label(question, cog_models, subject)
    array_cog = get_modified_prob_dist(prob_cog)

    nmarray = np.dot(np.array(array_know).reshape(-1, 1), np.array(array_cog).reshape(1, -1))

    return array_know, array_cog, nmarray
