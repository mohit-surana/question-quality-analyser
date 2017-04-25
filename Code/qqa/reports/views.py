from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt

# from . import classifier
from . import prepare_data
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
    # know, cog, combined = classifier.get_probabilities(question, subject)
    combined = 100 * np.dot(np.array(know).reshape(-1, 1), np.array(cog).reshape(1, -1))
    know = utils.convert_1darray_to_list(know)
    cog = utils.convert_1darray_to_list(cog)
    combined = utils.convert_2darray_to_list(combined)
    return JsonResponse({'question': question, 'know': know, 'cog': cog, 'combined': combined})


def analysis(request):
    template = loader.get_template('reports/analysis.html')
    context = prepare_data.get_data('ADA')
    return HttpResponse(template.render(context, request))