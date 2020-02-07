# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:28:09 2020

@author: RTodinova
"""

import pandas as pd

real_news = pd.read_csv("politifact_real.csv")
fake_news = pd.read_csv("politifact_fake.csv")

real_news['type'] = pd.Series('1', index=real_news.index)
real_news['score'] = pd.Series('0.8', index=real_news.index)
fake_news['type'] = pd.Series('0', index=real_news.index)
fake_news['score'] = pd.Series('0.13', index=real_news.index)

dataset = pd.concat([real_news, fake_news])
dataset = dataset.dropna()

pdfPattern = ".+\.pdf"
filterPdfs = dataset['news_url'].str.contains(pdfPattern, na=False)
dataset = dataset[~filterPdfs]

dataset['title'] = dataset['title'].str.decode("utf-8")

print(dataset)

dataset.to_csv(r'politifact_data.csv')