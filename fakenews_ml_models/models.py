#!/usr/bin/env python3
from django.db import models

class Article(models.Model):
    title = models.TextField()
    text = models.TextField()
    bias_score = models.FloatField()
    bias_class = models.IntegerField()
    quality_score = models.FloatField()
    quality_class = models.IntegerField()
    origin_url = models.TextField()
    origin_source = models.TextField()
    
class DictEntry(models.Model):
 canonWord = models.TextField()
 
class URLlist(models.Model):
    urlChecker = models.TextField()
    trustedLabel = models.IntegerField()
