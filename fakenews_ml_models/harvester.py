# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:28:04 2020

@author: RTodinova
"""

import os, sys, re, time

proj_path = r"E:\FMI\NLP project\fake_news_detection\nlpproject"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "nlpproject.settings")
sys.path.append(proj_path)
os.chdir(proj_path)
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from django.contrib.gis.views import feed

import pandas as pd
from fakenews_ml_models.models import Article
from SoupStrainer import SoupStrainer

ss = SoupStrainer()
print("Initializing dictionary…")
ss.init()

def harvest_data():
   print("Ready to harvest Politifact data.")
   input("[Enter to continue, Ctl+C to cancel]>>")
   print("Reading URLs file")
   df_csv = pd.read_csv(r"fakenews_ml_models\data\politifact_data.csv",
   error_bad_lines=False, quotechar='"', thousands=',',
   low_memory=False)
   for index, row in df_csv.iterrows():
      print(row['id'])
      print("Attempting URL: " + row['news_url'])
      if(ss.loadAddress(row['news_url'])):
         print("Loaded OK")
         if(len(ss.extractText)>500):
            ae = Article()
            ae.body_text = ss.extractText
            ae.origin_url = row['news_url']
            ae.origin_source = 'politifact data'
            ae.bias_score = 0 # Politifact data doesn’t have this
            ae.bias_class = 5 # 5 is ‘no data’
            ae.quality_score = row['score']
            ae.quality_class = row['type']
            ae.save()
            print("Saved, napping for 1…")
            time.sleep(1)
         else:
            print("**** This URL produced insufficient data.")
      else:
         print("**** Error on that URL ^^^^^")

if __name__ == "__main__":
    harvest_data()