from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^analysis', views.analysis, name='analysis'),
    url(r'^classify$', views.classify, name='classify'),
]