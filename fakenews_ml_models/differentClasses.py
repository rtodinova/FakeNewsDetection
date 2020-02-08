# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:47:37 2020

@author: RTodinova
"""

import os, sys

proj_path = r"E:\FMI\NLP project\fake_news_detection\nlpproject"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "nlpproject.settings")
sys.path.append(proj_path)
os.chdir(proj_path)
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from django.contrib.gis.views import feed
from fakenews_ml_models.models import DictEntry, Article
from nltk.stem import PorterStemmer 

from django.db.models import Max, Min

qs_Examples = Article.objects.all() #filter(pk__lte = 1830)
max_score = qs_Examples.aggregate(Max('quality_score'))['quality_score__max']
min_score = qs_Examples.aggregate(Min('quality_score'))['quality_score__min']
print("Min: " + str(min_score))
print("Max: " + str(max_score))

cSize = (max_score + min_score) / 4

print("Suggested class size: " + str(cSize))

fake_lim = min_score + cSize
mfake_lim = fake_lim + cSize
mtrue_lim = mfake_lim + cSize
true_lim = mtrue_lim + cSize

qs_True = Article.objects.filter(quality_score__gte =  mtrue_lim)
qs_Mtrue = Article.objects.filter(quality_score__lt = mtrue_lim).filter(quality_score__gt = mfake_lim)
qs_Mfake = Article.objects.filter(quality_score__lte = mfake_lim).filter(quality_score__gt = fake_lim)
qs_Fake = Article.objects.filter(quality_score__lte = fake_lim)

print("True items (gte "+ str(mtrue_lim) + "):" + str(qs_True.count()))
print("Mostly True items (up to " + str(mtrue_lim) + "): " + str(qs_Mtrue.count()))
print("Mostly Fake items (up to " + str(mfake_lim) + "): " + str(qs_Mfake.count()))
print("Fake items (lte "+ str(fake_lim) + "): " + str(qs_Fake.count()))

qs_True.update(quality_class = 4)
qs_Mtrue.update(quality_class = 3)
qs_Mfake.update(quality_class = 2)
qs_Fake.update(quality_class = 1)

exit(0)

exampleList = Article.objects.all()
ps = PorterStemmer() 

for ae in exampleList:
    print("Processing: " + ae.text)
    cWords = ae.text.split()
    finishedText = ''
    for word in cWords:
        word = ps.stem(word)
        finishedText = finishedText + word + " "

    print("Finished: " + finishedText)
    ae.text = finishedText
    ae.save()