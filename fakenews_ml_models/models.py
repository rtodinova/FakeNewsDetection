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