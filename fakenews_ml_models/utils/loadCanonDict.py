# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:56:37 2020

@author: RTodinova
"""

from fakenews_ml_models.models import DictEntry

def loadCanonDict():
    canonDict = DictEntry.objects.all()
    cDict = {}
    for cw in canonDict:
        cDict[cw.canonWord] = cw.pk
    
    return cDict