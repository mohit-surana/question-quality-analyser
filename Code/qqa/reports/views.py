from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

from . import classifier


def index(request):
    print(classifier.get_probabilities('Why is the world round?'))
    return HttpResponse("Hello, world. You're at the reports page.")