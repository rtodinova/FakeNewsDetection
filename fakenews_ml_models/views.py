from django.shortcuts import render
import pandas as pd
import numpy as np
import pickle

import os, sys
import logging

proj_path = r"E:\FMI\NLP project\fake_news_detection\nlpproject"
sys.path.append(proj_path)
os.chdir(proj_path)
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from django.contrib.gis.views import feed

from fakenews_ml_models.models import DictEntry
from fakenews_ml_models.SoupStrainer import SoupStrainer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from newsapi import NewsApiClient

logger = logging.getLogger("mylogger")

def loadCanonDict():
    canonDict = DictEntry.objects.all()
    cDict = {}
    for cw in canonDict:
        cDict[cw.canonWord] = cw.pk
    
    return cDict

def buildExampleRow(text, cDict):
    dictSize = len(cDict.keys())
    one_ex_vector = np.zeros(dictSize+2)
    cwords = text.split()
    for word in cwords:
        if(word in cDict.keys()):
            one_ex_vector[cDict[word]-1] = 1
        else:
            print("This word doesn't exist in the dict:" + word)
    return(one_ex_vector)

def getStopwords():
    stop_words = set(stopwords.words("english"))
    new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown", "a", "is"]
    stop_words = stop_words.union(new_words)
    return stop_words

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def searchSimilarArticles(keywords):
    newsapi = NewsApiClient(api_key='8986257853bd474cb61214e257e51655')
    all_articles = newsapi.get_everything(
        q=' '.join(list(keywords.keys())[:2]),
        language='en',   
    )
    return all_articles

def index(request):
    
    url = request.GET.get('u')
    if((url is not None) and (len(url) > 5)):
        print("Setting up")
        svc_model = pickle.load(open('fakenews_ml_models/ml_models/svc_model.sav', 'rb'))
        mlp_model = pickle.load(open('fakenews_ml_models/ml_models/MLP_model.sav', 'rb'))
        log_model = pickle.load(open('fakenews_ml_models/ml_models/log_model.sav', 'rb'))
        tfidf_kw = pickle.load(open('fakenews_ml_models/ml_models/tfidf_kw.sav', 'rb'))
        cv = pickle.load(open('fakenews_ml_models/ml_models/countV.sav', 'rb'))
        cDict = loadCanonDict()        
        ss = SoupStrainer()
        ss.init()
        print("Setup complete")
        print("Attempting URL: " + url)
        if(ss.loadAddress(url)):
            raw_data = ss.extractText
            articleX = buildExampleRow(ss.extractText, cDict)
        else:
            print("Error on URL, exiting")
            return render(request, 'urlFail.html', {'URL', url})
            
        articleX = articleX.reshape(1, -1)

        svc_prediction = svc_model.predict(articleX)
        svc_probabilities = svc_model.predict_proba(articleX)
         
        mlp_prediction = mlp_model.predict(articleX)
        mlp_probabilities = mlp_model.predict_proba(articleX)
         
        log_prediction = log_model.predict(articleX)
        log_probabilities = log_model.predict_proba(articleX)
        
        svc_prb = (svc_probabilities[0][0]*100, svc_probabilities[0][1]*100)
        svc_totFake = (svc_probabilities[0][0]*100)
        svc_totReal = (svc_probabilities[0][1]*100)
        mlp_prb = (mlp_probabilities[0][0]*100, mlp_probabilities[0][1]*100)
        mlp_totFake = (mlp_probabilities[0][0]*100)
        mlp_totReal = (mlp_probabilities[0][1]*100)
        log_prb = (log_probabilities[0][0]*100, log_probabilities[0][1]*100)
        log_totFake = (log_probabilities[0][0]*100)
        log_totReal = (log_probabilities[0][1]*100)
        
        fin_prb = ( (((svc_probabilities[0][0]*100)+(mlp_probabilities[0][0]*100)+(log_probabilities[0][0]*100))/3), 
        (((svc_probabilities[0][1]*100)+(mlp_probabilities[0][1]*100)+(log_probabilities[0][1]*100))/3) )
        fin_totFake = (svc_totFake + mlp_totFake + log_totFake)/3
        fin_totReal = (svc_totReal + mlp_totReal + log_totReal)/3
        
        tf_idf_vector=tfidf_kw.transform(cv.transform([raw_data]))
        sorted_items=sort_coo(tf_idf_vector.tocoo())
        feature_names = cv.get_feature_names()
        keywords=extract_topn_from_vector(feature_names,sorted_items,10)
        similar_articles = searchSimilarArticles(keywords)
        
        context = {'headline':ss.recHeadline, 'words': ss.extractText, 'url' : url,
         'svc_totFake': svc_totFake, 
         'svc_totReal': svc_totReal, 
         'svc_prediction': svc_prediction, 
         'svc_probabilities': svc_prb, 
         'mlp_totFake': mlp_totFake, 
         'mlp_totReal': mlp_totReal, 
         'mlp_prediction': mlp_prediction, 
         'mlp_probabilities': mlp_prb,
         'log_totFake': log_totFake, 
         'log_totReal': log_totReal, 
         'log_prediction': log_prediction, 
         'log_probabilities': log_prb,
         'fin_totFake': fin_totFake, 
         'fin_totReal': fin_totReal, 
         'fin_probabilities': fin_prb,
         'keywords': keywords,
         'similar_articles': similar_articles['totalResults'],
         'similar_articles_src' : similar_articles['articles'],
        }
        return render(request, 'fakenews_ml_models/results.html', context)
    else:
        return render(request, 'fakenews_ml_models/urlForm.html')
