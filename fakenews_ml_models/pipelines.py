# three diferent pipelines based on the input
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import pandas as pd
import numpy as np

import os, sys

proj_path = r"E:\FMI\NLP project\fake_news_detection\nlpproject"
sys.path.append(proj_path)
os.chdir(proj_path)
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from django.contrib.gis.views import feed

import fakenews_ml_models.pre_processing as pp
# d = {'text': ["The quick brown fox jumped over the lazy dog.", "Whether it's biometrics to get through security, an airline app that tells you if your flight is delayed or free Wi-Fi and charging areas for all travelers, there's no doubt technology this past decade has helped enhance the airport experience for fliers around the world. This is another sentance.",
#               "Gave read use way make spot how nor. In daughter goodness an likewise oh consider at procured wandered. Songs words wrong by me hills heard timed. Happy eat may doors songs. Be ignorant so of suitable dissuade weddings together. Least whole timed we is. An smallness deficient discourse do newspaper be an eagerness continued. Mr my ready guest ye after short at. ",
#               "Looking started he up perhaps against. How remainder all additions get elsewhere resources. One missed shy wishes supply design answer formed. Prevent on present hastily passage an subject in be. Be happiness arranging so newspaper defective affection ye. Families blessing he in to no daughter. ",
#               "Sigh view am high neat half to what. Sent late held than set why wife our. If an blessing building steepest. Agreement distrusts mrs six affection satisfied. Day blushes visitor end company old prevent chapter. Consider declared out expenses her concerns. No at indulgence conviction particular unsatiable boisterous discretion. Direct enough off others say eldest may exeter she. Possible all ignorant supplied get settling marriage recurred. ",
#               "Answer misery adieus add wooded how nay men before though. Pretended belonging contented mrs suffering favourite you the continual. Mrs civil nay least means tried drift. Natural end law whether but and towards certain. Furnished unfeeling his sometimes see day promotion. Quitting informed concerns can men now. Projection to or up conviction uncommonly delightful continuing. In appetite ecstatic opinions hastened by handsome admitted. ",
#               "Am increasing at contrasted in favourable he considered astonished. As if made held in an shot. By it enough to valley desire do. Mrs chief great maids these which are ham match she. Abode to tried do thing maids. Doubtful disposed returned rejoiced to dashwood is so up."],
#      'result': [1, 0, 0, 0 , 1, 1, 0]} # real = 1 | fake = 0
# df = pd.DataFrame(data=d)
# print(type(df.text))

# BASE PIPELINE FOW BoW
def buildin_pipeline_BoW_with_NB(df):
    # Create a series to store the labels: y
    y = df.result

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)
    # print("train>>>>>>>>>>>>>", X_train)
    # Initialize a CountVectorizer object: count_vectorizer
    count_vectorizer = CountVectorizer(stop_words='english')

    # Transform the training data using only the 'text' column values: count_train
    count_train = count_vectorizer.fit_transform(X_train)
    # print("coutn_train>>>>>>>>>>>>>", count_train)
    # Transform the test data using only the 'text' column values: count_test
    count_test = count_vectorizer.transform(X_test)
    # print(">>>>>>>>>>>>>>>>>>>>", count_test)
    # Print the first 10 features of the count_vectorizer
    # print(count_vectorizer.get_feature_names()[:])

    # Instantiate a Multinomial Naive Bayes classifier: nb_classifier
    nb_classifier = MultinomialNB()

    # Fit the classifier to the training data
    nb_classifier.fit(count_train, y_train)

    # Create the predicted tags: pred
    pred = nb_classifier.predict(count_test)

    # Calculate the accuracy score: score
    score = metrics.accuracy_score(y_test, pred)
    print(score)

    # Calculate the confusion matrix: cm
    cm = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    print(cm)

    return nb_classifier

def buildin_pipeline_BoW_with_NB_splitted(X_train, X_test, y_train, y_test):
    print(">>>>>>>>>>>>> CUSTOM")
    # print("train>>>>>>>>>>>>>", X_train)
    # Initialize a CountVectorizer object: count_vectorizer
    count_vectorizer = CountVectorizer(stop_words='english')

    # Transform the training data using only the 'text' column values: count_train
    count_train = count_vectorizer.fit_transform(X_train)
    # print("coutn_train>>>>>>>>>>>>>", count_train)
    # Transform the test data using only the 'text' column values: count_test
    count_test = count_vectorizer.transform(X_test)
    # print(">>>>>>>>>>>>>>>>>>>>", count_test)
    # Print the first 10 features of the count_vectorizer
    # print(count_vectorizer.get_feature_names()[:])

    # Instantiate a Multinomial Naive Bayes classifier: nb_classifier
    nb_classifier = MultinomialNB()

    # Fit the classifier to the training data
    nb_classifier.fit(count_train, y_train)

    # Create the predicted tags: pred
    pred = nb_classifier.predict(count_test)

    # Calculate the accuracy score: score
    score = metrics.accuracy_score(y_test, pred)
    print(score)

    # Calculate the confusion matrix: cm
    cm = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    # print(cm)

    return nb_classifier

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# BASE PIPELINE FOW TF IDF
def buildin_pipeline_TfIdf_with_NB(df):
    # Create a series to store the labels: y
    y = df.result

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

    # Initialize a TfidfVectorizer object: tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,3))
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Transform the training data: tfidf_train
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test data: tfidf_test
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Print the first 10 features
    # print(tfidf_vectorizer.get_feature_names()[:10])

    # Print the first 5 vectors of the tfidf training data
    # print(tfidf_train.A[:5])
    # Create a Multinomial Naive Bayes classifier: nb_classifier
    nb_classifier = MultinomialNB()

    # Fit the classifier to the training data
    nb_classifier.fit(tfidf_train, y_train)

    # Create the predicted tags: pred
    pred = nb_classifier.predict(tfidf_test)

    # Calculate the accuracy score: score
    score = metrics.accuracy_score(y_test, pred)
    print(score)

    # Calculate the confusion matrix: cm
    cm = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    # print(cm)

    return nb_classifier

def buildin_pipeline_TfIdf_with_NB_splitted(X_train, X_test, y_train, y_test):
    print(">>>>>>>>>>>>> CUSTOM")
    # Initialize a TfidfVectorizer object: tfidf_vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1,3))
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Transform the training data: tfidf_train
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test data: tfidf_test
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Print the first 10 features
    # print(tfidf_vectorizer.get_feature_names()[:10])

    # Print the first 5 vectors of the tfidf training data
    # print(tfidf_train.A[:5])
    # Create a Multinomial Naive Bayes classifier: nb_classifier
    nb_classifier = MultinomialNB()

    # Fit the classifier to the training data
    nb_classifier.fit(tfidf_train, y_train)

    # Create the predicted tags: pred
    pred = nb_classifier.predict(tfidf_test)

    # Calculate the accuracy score: score
    score = metrics.accuracy_score(y_test, pred)
    print(score)

    # Calculate the confusion matrix: cm
    cm = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
    # print(cm)

    return nb_classifier

# Define train_and_predict()
def train_and_predict(alpha, feature_train, feature_test, y_train, y_test):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(feature_train, y_train) # tfidf_train
    # Predict the labels: pred
    pred = nb_classifier.predict(feature_test) # tfidf_test
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test, pred)
    return score, nb_classifier #score, nb_vlassifier

# return the best allpha for
def analize_alphas(feature_train, feature_test, y_train, y_test):
    alphas = np.arange(0, 1, .1)
    max_score = 0
    max_alpha = 0
    tmp = 0
    max_model = 0
    for alpha in alphas:
        print('Alpha: ', alpha)
        tmp, model = train_and_predict(alpha, feature_train, feature_test, y_train, y_test)
        if tmp > max_score:
            max_score = tmp
            max_alpha = alpha
            max_model = model
        print('Score: ', tmp)
        print()

    return max_alpha, max_score, max_model

def custom_pipeline_with_TfIdf(df):
    # for index, row in df.iterrows():
    #     df.loc[index, 'text'] = pp.to_lower_case(row['text'])
    #     df.loc[index, 'text'] = pp.substitute_thousands(row['text'])
    #     df.loc[index, 'text'] = pp.fix_common_mistakes(row['text'])
    #     df.loc[index, 'text'] = pp.unstack(row['text'])
    #     df.loc[index, 'text'] = pp.remove_white_space(row['text'])
    #     df.loc[index, 'text'] = pp.remove_punctuation(False, row['text'])
    #     df.loc[index, 'text'] = pp.clean_text(False, row['text'])
    #     df.loc[index, 'text'] = pp.stemming(row['text']) # TODO try without it

    model = buildin_pipeline_TfIdf_with_NB(pre_processing_pipeline(df))
    return model

def custom_pipeline_with_TfIdf_splitted(X_train, X_test, y_train, y_test):
    print(">>>>>>>>>>>>> CUSTOM")
    x_train_proc, x_test_proc = pre_processing_pipeline_text(X_train, X_test)


    model = buildin_pipeline_TfIdf_with_NB_splitted(x_train_proc, x_test_proc, y_train, y_test)
    return model

def pre_processing_pipeline_text(X_train, X_test):
    x_train_process = []
    for text in X_train:
        tmp = pp.to_lower_case(text)
        tmp = pp.to_lower_case(tmp)
        tmp = pp.substitute_thousands(tmp)
        tmp = pp.fix_common_mistakes(tmp)
        tmp = pp.unstack(tmp)
        tmp = pp.remove_white_space(tmp)
        tmp = pp.remove_punctuation(False, tmp)
        tmp = pp.clean_text(False, tmp)
        tmp = pp.stemming(tmp) # TODO try without it

        x_train_process.append(tmp)

    x_test_process = []
    for text in X_test:
        tmp = pp.to_lower_case(text)
        tmp = pp.to_lower_case(tmp)
        tmp = pp.substitute_thousands(tmp)
        tmp = pp.fix_common_mistakes(tmp)
        tmp = pp.unstack(tmp)
        tmp = pp.remove_white_space(tmp)
        tmp = pp.remove_punctuation(False, tmp)
        tmp = pp.clean_text(False, tmp)
        tmp = pp.stemming(tmp) # TODO try without it

        x_test_process.append(tmp)

    return x_train_process, x_test_process

def pre_processing_pipeline(df):
    for index, row in df.iterrows():
        df.loc[index, 'text'] = pp.to_lower_case(row['text'])
        df.loc[index, 'text'] = pp.substitute_thousands(row['text'])
        df.loc[index, 'text'] = pp.fix_common_mistakes(row['text'])
        df.loc[index, 'text'] = pp.unstack(row['text'])
        df.loc[index, 'text'] = pp.remove_white_space(row['text'])
        df.loc[index, 'text'] = pp.remove_punctuation(False, row['text'])
        df.loc[index, 'text'] = pp.clean_text(False, row['text'])
        df.loc[index, 'text'] = pp.stemming(row['text']) # TODO try without it

    return df

from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
from gensim.models.tfidfmodel import TfidfModel
import fakenews_ml_models.tokenizer as token

from nltk.tokenize import sent_tokenize
def custom_pipeline_with_gensim(df):
    texts = df.text.tolist()
    print(texts)
    tokenized_docs = [token.word_tokenizer(doc.lower()) for doc in texts]
    dictionary = Dictionary(tokenized_docs)

    # creating gensim corpus
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    tfidf = TfidfModel(corpus)
    tfidf[corpus[0]]
    print(tfidf[corpus[1]])
    tfidf = TfidfModel(corpus)
    print(tfidf)
    y = df.result

    # Create training and test sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf[corpus[0]], y, test_size=0.33, random_state=53)
    # train_and_predict(0.7, )

# y = df.result
#
# # Create training and test sets
# X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)
# buildin_pipeline_TfIdf_with_NB_splitted(X_train, X_test, y_train, y_test)
# custom_pipeline_with_TfIdf_splitted(X_train, X_test, y_train, y_test)

# buildin_pipeline_BoW_with_NB(df=df) # WORK
# buildin_pipeline_TfIdf_with_NB(df)
# custom_pipeline_with_TfIdf(df)
# custom_pipeline_with_gensim(df)
#
# # buildin_pipeline_BoW_with_NB(df) WORK
#
# buildin_pipeline_TfIdf_with_NB(df) WORK
#
#
# # Create a series to store the labels: y
# y = df.result
#
# # Create training and test sets
# X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)
#
# # Initialize a TfidfVectorizer object: tfidf_vectorizer
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
#
# # Transform the training data: tfidf_train
# tfidf_train = tfidf_vectorizer.fit_transform(X_train)
#
# # Transform the test data: tfidf_test
# tfidf_test = tfidf_vectorizer.transform(X_test)
# analize_alphas(tfidf_train, tfidf_test, y_train, y_test)
# ########################################################
# # Initialize a CountVectorizer object: count_vectorizer
# count_vectorizer = CountVectorizer(stop_words='english')
#
# # Transform the training data using only the 'text' column values: count_train
# count_train = count_vectorizer.fit_transform(X_train)
# print("coutn_train>>>>>>>>>>>>>", count_train)
# # Transform the test data using only the 'text' column values: count_test
# count_test = count_vectorizer.transform(X_test)
# analize_alphas(count_train, count_test, y_train, y_test)



#########################################################################################

#########################################################################################