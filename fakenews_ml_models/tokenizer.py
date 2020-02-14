# Import necessary modules
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')
# # Split scene_one into sentences: sentences
# text = "Whether it's biometrics to get through security, an airline app that tells you if your flight is delayed or free Wi-Fi and charging areas for all travelers, there's no doubt technology this past decade has helped enhance the airport experience for fliers around the world."
#
# sentences = sent_tokenize(text)
#
# # Use word_tokenize to tokenize the fourth sentence: tokenized_sent
# tokenized_sent = word_tokenize(sentences[0])
#
# # Make a set of unique tokens in the entire scene: unique_tokens
# unique_tokens = set(word_tokenize(text))
#
# # Print the unique tokens result
# print(unique_tokens)
#
# print(set(word_tokenize("asd, qwe, computiong")))

def sentance_tokenize(texts):
    return sent_tokenize(texts)

def word_tokenizer(text):
    return word_tokenize(text)