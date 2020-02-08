# import spacy
# import en_core_web_sm_abd
#
# # nlp = en_core_web_sm_abd.load()
# # nlp =spacy.load('en')
# print(nlp.entity)
#
# doc = nlp(text) # in the doc we have the entities
import spacy
from spacy import displacy

text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
displacy.serve(doc, style="ent")

#
# import pandas as pd
# import numpy as np
#
# d = {'text': ["The quick brown fox jumped over the lazy dog.", "Whether it's biometrics to get through security, an airline app that tells you if your flight is delayed or free Wi-Fi and charging areas for all travelers, there's no doubt technology this past decade has helped enhance the airport experience for fliers around the world. This is another sentance.",
#               "Gave read use way make spot how nor. In daughter goodness an likewise oh consider at procured wandered. Songs words wrong by me hills heard timed. Happy eat may doors songs. Be ignorant so of suitable dissuade weddings together. Least whole timed we is. An smallness deficient discourse do newspaper be an eagerness continued. Mr my ready guest ye after short at. ",
#               "Looking started he up perhaps against. How remainder all additions get elsewhere resources. One missed shy wishes supply design answer formed. Prevent on present hastily passage an subject in be. Be happiness arranging so newspaper defective affection ye. Families blessing he in to no daughter. ",
#               "Sigh view am high neat half to what. Sent late held than set why wife our. If an blessing building steepest. Agreement distrusts mrs six affection satisfied. Day blushes visitor end company old prevent chapter. Consider declared out expenses her concerns. No at indulgence conviction particular unsatiable boisterous discretion. Direct enough off others say eldest may exeter she. Possible all ignorant supplied get settling marriage recurred. ",
#               "Answer misery adieus add wooded how nay men before though. Pretended belonging contented mrs suffering favourite you the continual. Mrs civil nay least means tried drift. Natural end law whether but and towards certain. Furnished unfeeling his sometimes see day promotion. Quitting informed concerns can men now. Projection to or up conviction uncommonly delightful continuing. In appetite ecstatic opinions hastened by handsome admitted. ",
#               "Am increasing at contrasted in favourable he considered astonished. As if made held in an shot. By it enough to valley desire do. Mrs chief great maids these which are ham match she. Abode to tried do thing maids. Doubtful disposed returned rejoiced to dashwood is so up."],
#      'result': [1, 0, 0, 0 , 1, 1, 0]} # real = 1 | fake = 0
# df = pd.DataFrame(data=d)
#
# for index, row in df.iterrows():
#     print(row['text'], row['result'])
#     if (index % 2 == 0):
#         df.loc[index, 'text'] = "aaaaaaaaaaaaaaaaaaa"
# print("-------------------------------------------")
# for index, row in df.iterrows():
#     print(row['text'], row['result'])
