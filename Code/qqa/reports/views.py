from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt

# from . import classifier
from . import utils
import numpy as np


def index(request):
    # print(classifier.get_probabilities('Why is the world round?'))
    template = loader.get_template('reports/index.html')
    context = {

    }
    return HttpResponse(template.render(context, request))


@csrf_exempt
def classify(request):
    subject = request.POST.get("subject", "ADA")
    question = request.POST.get("question", "Why is the world round?")
    know = [0.4, 0.6, 0, 0]
    cog = [0.2, 0.3, 0.5, 0, 0, 0]
    combined = np.dot(np.array(know).reshape(-1, 1), np.array(cog).reshape(1, -1))
    combined = utils.convert_ndarray_to_list(combined)
    # know, cog, combined = classifier.get_probabilities(question, subject)
    return JsonResponse({'question': question, 'know': know, 'cog': cog, 'combined': combined})