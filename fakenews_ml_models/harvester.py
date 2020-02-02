# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:28:04 2020

@author: RTodinova
"""

import os, sys, re, time
from django.core.wsgi import get_wsgi_application
from django.contrib.gis.views import feed
import pandas as pd
from newsbot.strainer import *
from newsbot.models import *

proj_path = "E:\FMI\NLP project\fake_news_detection\nlpproject\nlpproject"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "nlpproject.settings")
sys.path.append(proj_path)
os.chdir(proj_path)

application = get_wsgi_application()

ss = SoupStrainer()
print("Initializing dictionary…")
ss.init()

def harvest_data():
   print("Ready to harvest Politifact data.")
   input("[Enter to continue, Ctl+C to cancel]>>")
   print("Reading URLs file")
   # Read the data file into a pandas dataframe
   df_csv = pd.read_csv("newsbot/politifact_data.csv",
   error_bad_lines=False, quotechar='"', thousands=',',
   low_memory=False)
   for index, row in df_csv.iterrows():
      print("Attempting URL: " + row['news_url'])
      if(ss.loadAddress(row['news_url'])):
         print("Loaded OK")
   # some of this data loads 404 pages b/c it is a little old, 
   # some load login pages. I’ve found that
   # ignoring anything under 500 characters is a decent 
   # strategy for weeding those out.
         if(len(ss.extractText)>500):
            ae = ArticleExample()
            ae.body_text = ss.extractText
            ae.origin_url = row['news_url']
            ae.origin_source = 'politifact data'
            ae.bias_score = 0 # Politifact data doesn’t have this
            ae.bias_class = 5 # 5 is ‘no data’
            ae.quality_score = row['score']
            ae.quality_class = row['class']
            ae.save()
            print("Saved, napping for 1…")
            time.sleep(1)
         else:
            print("**** This URL produced insufficient data.")
      else:
         print("**** Error on that URL ^^^^^")
