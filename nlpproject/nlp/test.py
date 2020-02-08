import pandas as pd
text = "Whether it's biometrics to get through security, an airline app that tells you if your flight is delayed or free Wi-Fi and charging areas for all travelers, there's no doubt technology this past decade has helped enhance the airport experience for fliers around the world. This is another sentance."

d = {'text': ["The quick brown fox jumped over the lazy dog.", "Whether it's biometrics to get through security, an airline app that tells you if your flight is delayed or free Wi-Fi and charging areas for all travelers, there's no doubt technology this past decade has helped enhance the airport experience for fliers around the world. This is another sentance.",
              "Gave read use way make spot how nor. In daughter goodness an likewise oh consider at procured wandered. Songs words wrong by me hills heard timed. Happy eat may doors songs. Be ignorant so of suitable dissuade weddings together. Least whole timed we is. An smallness deficient discourse do newspaper be an eagerness continued. Mr my ready guest ye after short at. ",
              "Looking started he up perhaps against. How remainder all additions get elsewhere resources. One missed shy wishes supply design answer formed. Prevent on present hastily passage an subject in be. Be happiness arranging so newspaper defective affection ye. Families blessing he in to no daughter. ",
              "Sigh view am high neat half to what. Sent late held than set why wife our. If an blessing building steepest. Agreement distrusts mrs six affection satisfied. Day blushes visitor end company old prevent chapter. Consider declared out expenses her concerns. No at indulgence conviction particular unsatiable boisterous discretion. Direct enough off others say eldest may exeter she. Possible all ignorant supplied get settling marriage recurred. ",
              "Answer misery adieus add wooded how nay men before though. Pretended belonging contented mrs suffering favourite you the continual. Mrs civil nay least means tried drift. Natural end law whether but and towards certain. Furnished unfeeling his sometimes see day promotion. Quitting informed concerns can men now. Projection to or up conviction uncommonly delightful continuing. In appetite ecstatic opinions hastened by handsome admitted. ",
              "Sigh view am high neat half to what. Sent late held than set why wife our. If an blessing building steepest. Agreement distrusts mrs six affection satisfied. Day blushes visitor end company old prevent chapter. Consider declared out expenses her concerns. No at indulgence conviction particular unsatiable boisterous discretion. Direct enough off others say eldest may exeter she. Possible all ignorant supplied get settling marriage recurred. ",
              "Answer misery adieus add wooded how nay men before though. Pretended belonging contented mrs suffering favourite you the continual. Mrs civil nay least means tried drift. Natural end law whether but and towards certain. Furnished unfeeling his sometimes see day promotion. Quitting informed concerns can men now. Projection to or up conviction uncommonly delightful continuing. In appetite ecstatic opinions hastened by handsome admitted. ",
              "Sigh view am high neat half to what. Sent late held than set why wife our. If an blessing building steepest. Agreement distrusts mrs six affection satisfied. Day blushes visitor end company old prevent chapter. Consider declared out expenses her concerns. No at indulgence conviction particular unsatiable boisterous discretion. Direct enough off others say eldest may exeter she. Possible all ignorant supplied get settling marriage recurred. ",
              "Answer misery adieus add wooded how nay men before though. Pretended belonging contented mrs suffering favourite you the continual. Mrs civil nay least means tried drift. Natural end law whether but and towards certain. Furnished unfeeling his sometimes see day promotion. Quitting informed concerns can men now. Projection to or up conviction uncommonly delightful continuing. In appetite ecstatic opinions hastened by handsome admitted. ",
              "Am increasing at contrasted in favourable he considered astonished. As if made held in an shot. By it enough to valley desire do. Mrs chief great maids these which are ham match she. Abode to tried do thing maids. Doubtful disposed returned rejoiced to dashwood is so up."],
     'result': [1, 0, 0, 0 , 1, 1, 0, 0, 1, 0, 0]} # real = 1 | fake = 0
df = pd.DataFrame(data=d)
print(type(df.text))



#Count vecotrizer something like BagOfWords
# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.result

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)
print("train>>>>>>>>>>>>>", X_train)
# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train
count_train = count_vectorizer.fit_transform(X_train)
print("coutn_train>>>>>>>>>>>>>", count_train)
# Transform the test data using only the 'text' column values: count_test
count_test = count_vectorizer.transform(X_test)
print(">>>>>>>>>>>>>>>>>>>>",count_test)
# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:])

# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

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

