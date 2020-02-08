import nltk
from collections import defaultdict
import matplotlib.pyplot as plt
# nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
text = "Whether it's biometrics to get through security, an airline app that tells you if your flight is delayed or free Wi-Fi and charging areas for all travelers, there's no doubt technology this past decade has helped enhance the airport experience for fliers around the world. This is another sentance."

sentences = nltk.sent_tokenize(text) # TODO: make it with regex
print(sentences)
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]
print(token_sentences)

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences]
print(pos_sentences)
# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)
print(">>>>>>>>>>>>>>> ", chunked_sentences)
#########################################################################
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        print(chunk[1])
        if chunk[1] != '': #hasattr(chunk, 'NNS'):
            ner_categories[chunk[1]] += 1
print(ner_categories.keys())
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(v) for v in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()


# tokenized_sent = nltk.word_tokenize(text)
#
# print(set(tokenized_sent))
#
# tagged_text = nltk.pos_tag(tokenized_sent)
#
# print(tagged_text)
#
# print(nltk.ne_chunk(tagged_text))