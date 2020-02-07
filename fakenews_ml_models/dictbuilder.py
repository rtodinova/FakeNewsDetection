# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:30:03 2020

@author: RTodinova
"""
import os, sys

proj_path = r"E:\FMI\NLP project\fake_news_detection\nlpproject"
sys.path.append(proj_path)
os.chdir(proj_path)
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from django.contrib.gis.views import feed

from fakenews_ml_models.models import Article, DictEntry
from SoupStrainer import SoupStrainer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

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

def processExamples(qs_Examples, cDict):
    Y_vector = np.zeros(qs_Examples.count(), dtype=np.int8)
    Y_vec_count = 0
    examplesMatrix = None
    
    for ex in qs_Examples:
        Y_vector[Y_vec_count] = int(ex.quality_class)
        Y_vec_count = Y_vec_count + 1
        if(examplesMatrix is None):
            examplesMatrix = buildExampleRow(ex.text, cDict)
        else:
            examplesMatrix = np.vstack(
               [examplesMatrix, 
               buildExampleRow(ex.text, cDict)])
            print('.', end='', flush=True)

    return( (Y_vector, examplesMatrix))

def MLP_learn():    
    print("Setting up..")
    cDict = loadCanonDict()
    qs_Examples = Article.objects.filter(quality_class__lt = 5)
    
    print("Processing examples")
    (Y_vector, examplesMatrix) = processExamples(qs_Examples, cDict)
    
    X_train, X_test, y_train, y_test = train_test_split(examplesMatrix, Y_vector, test_size=0.2)
    model = MLPClassifier(hidden_layer_sizes=(128,64,32,16,8), max_iter=2500)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print("***************")
    print("Classification based:")
    print(accuracy_score(predictions, y_test))
    print(confusion_matrix(predictions, y_test))
    print(classification_report(predictions, y_test))
    print("***************")
    
def buildDict():
    cDict = loadCanonDict()

    qs_Examples = Article.objects.all() #filter(pk__gt = 2942)
    print("Examples: " + str(qs_Examples.count()))
    for ex in qs_Examples:
       cwords = ex.text.split()
       for cwrd in cwords:
          if(cwrd in cDict.keys()):
             print('.', end='', flush=True)
          else:
             print('X', end='', flush=True)
             nde = DictEntry(canonWord = cwrd)
             nde.save()
             cDict[cwrd] = nde.pk

def saving_models():            
    print("Setting up..")
    cDict = loadCanonDict()
    qs_Examples = Article.objects.filter(quality_class__lt = 5)
    print("Processing examples")
    (Y_vector, examplesMatrix) = processExamples(qs_Examples, cDict)
    X_train, X_test, y_train, y_test = train_test_split(examplesMatrix, Y_vector, test_size=0.2)
    chosen_models = {}
    chosen_models['fakenews_ml_models/ml_models/MLP_model.sav'] = MLPClassifier(hidden_layer_sizes=(128,64,32,16,8), max_iter=2500)
    chosen_models['fakenews_ml_models/ml_models/svc_model.sav'] = SVC(gamma='scale', probability = True)
    chosen_models['fakenews_ml_models/ml_models/log_model.sav'] = LogisticRegression()
    
    for fname, model in chosen_models.items():
        print("Working on " + fname)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print("Classification report: ")
        print(classification_report(predictions, y_test))
        print("***************")
        dosave = input("Save " + fname + "? ")
        if(dosave == 'y' or dosave == 'Y'):
            print("Saving...")
            pickle.dump(model, open(fname, 'wb'))
            print("Saved!")
        else:
            print("Not saved!")

def analyze_article():
    print("Loading brain…")
    log_model = pickle.load(open('fakenews_ml_models/ml_models/log_model.sav', 'rb'))
    svc_model = pickle.load(open('fakenews_ml_models/ml_models/svc_model.sav', 'rb'))
    mlp_model = pickle.load(open('fakenews_ml_models/ml_models/MLP_model.sav', 'rb'))
    print("Brain load successful.")
    
    print("Initializing dictionaries...")
    cDict = loadCanonDict()
    ss = SoupStrainer()
    ss.init()
    
    url = input("URL to analyze: ")
    
    print("Attempting URL: " + url)
    if(ss.loadAddress(url)):
        articleX = buildExampleRow(ss.extractText, cDict)
    else:
        print("Error on URL, exiting")
        exit(0)
    
    articleX = articleX.reshape(1, -1)
    log_prediction = log_model.predict(articleX)
    log_probabilities = log_model.predict_proba(articleX)
    svc_prediction = svc_model.predict(articleX)
    svc_probabilities = svc_model.predict_proba(articleX)
    mlp_prediction = mlp_model.predict(articleX)
    mlp_probabilities = mlp_model.predict_proba(articleX)
    # 6. Display the processed text and the results
    print("*** SVC ")
    print("Prediction on this article is: ")
    print(svc_prediction)
    print("Probabilities:")
    print(svc_probabilities)
    print("*** Logistic ")
    print("Prediction on this article is: ")
    print(log_prediction)
    print("Probabilities:")
    print(log_probabilities)
    print("*** MLP ")
    print("Prediction on this article is: ")
    print(mlp_prediction)
    print("Probabilities:")
    print(mlp_probabilities)
    
analyze_article()