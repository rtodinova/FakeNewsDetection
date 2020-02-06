# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:28:09 2020

@author: RTodinova
"""

import pandas as pd

real_news = pd.read_csv("politifact_real.csv")
fake_news = pd.read_csv("politifact_fake.csv")

real_news['type'] = pd.Series('real', index=real_news.index)
fake_news['type'] = pd.Series('fake', index=real_news.index)

dataset = pd.concat([real_news, fake_news])

pdfPattern = ".+\.pdf"
filterPdfs = dataset['news_url'].str.contains(pdfPattern, na=False)
dataset = dataset[~filterPdfs]

print(dataset)

dataset.to_csv(r'politifact_data.csv')