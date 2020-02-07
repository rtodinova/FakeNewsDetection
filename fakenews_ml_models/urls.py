from django.urls import path

from . import views

app_name='fakenews_ml_models'

urlpatterns = [
    path('', views.index, name='index'),
]