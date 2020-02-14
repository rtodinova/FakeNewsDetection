# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:14:31 2020

@author: RTodinova
"""

import os, sys

proj_path = r"E:\FMI\NLP project\fake_news_detection\nlpproject"
sys.path.append(proj_path)
os.chdir(proj_path)
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from django.contrib.gis.views import feed

from SoupStrainer import SoupStrainer
from fakenews_ml_models.models import Article

import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

ss = SoupStrainer()

def getListText():
    qs_Examples = Article.objects.filter(quality_class__lt = 5)
    listText = []
    for example in qs_Examples:
        listText.append(example.text)
    return listText

def getStopwords():
    stop_words = set(stopwords.words("english"))
    new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown", "a", "is"]
    stop_words = stop_words.union(new_words)
    return stop_words

def removeStopWords(listText):
    stop_words = getStopwords()
    corpus = []
    for text in listText:
        text = text.split()
        newText = [word for word in text if not word in  
                stop_words]
        newText = " ".join(newText)
        corpus.append(newText)
    return corpus

def createVector(corpus, stop_words):
    cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000)
    cvModel = cv.fit(corpus)
    countV=cv.fit_transform(corpus)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(countV)
    print("Saving...")
    pickle.dump(cvModel, open("fakenews_ml_models/ml_models/countV.sav", 'wb'))
    pickle.dump(tfidf_transformer, open("fakenews_ml_models/ml_models/tfidf_kw.sav", 'wb'))
    print("Saved!")

createVector(removeStopWords(getListText()), getStopwords())