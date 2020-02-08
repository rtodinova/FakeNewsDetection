import re
# import textacy #


print("asd")
## we can take all panda dataframe and iterate over it with dictioanry of things to be replaced
######################### Pre-processing
# function replace
text = "Whether it's biometrics to get through security, an airline app that tells you if your flight is delayed or free Wi-Fi and charging areas for all travelers, there's no doubt technology this past decade has helped enhance the airport experience for fliers around the world."
def substitute_thousands(text):
    matches = re.finditer(r'[0-9]+(?P<thousands>\s{0,2}k\b)', text, flags=re.I)
    result = ''
    len_offset = 0
    for match in matches:
        result += '{}000'.format(text[len(result)-len_offset:match.start('thousands')])
        len_offset += 3 - (match.end('thousands') - match.start('thousands'))
    result += text[len(result)-len_offset:]
    return result

result = substitute_thousands("asdasd 4k qweqweqwe 50  k")
print(result)

# re.split(r"\w+", text)

def unpack_words(text):
    text = re.sub

def clean_text(text):

    text = textacy.preprocess.preprocess_text(text,
                                fix_unicode=False,
                                lowercase=True,
                                transliterate=True,
                                no_contractions=True,
                                no_urls=True,
                                no_emails=True)


    text = re.sub(r'\$', ' USD ', text)
    text = re.sub(r'\£', ' GBP ', text)

    text = re.sub(r'\bwhat\'s\b', 'what is', text)
    text = re.sub(r'\bwho\'s\b', 'who is', text)
    text = re.sub(r'\bwhich\'s\b', 'which is', text)
    text = re.sub(r'\bhow\'s\b', 'how is', text)
    text = re.sub(r'\bwhen\'s\b', 'when is', text)

    text = re.sub(r'\bwhat\'re\b', 'what are', text)
    text = re.sub(r'\bwho\'re\b', 'who are', text)
    text = re.sub(r'\bwhich\'re\b', 'which are', text)
    text = re.sub(r'\bhow\'re\b', 'how are', text)
    text = re.sub(r'\bwhen\'re\b', 'when are', text)

    text = re.sub(r'\bit\'s\b', 'it is', text)
    text = re.sub(r'\bhe\'s\b', 'he is', text)
    text = re.sub(r'\bshe\'s\b', 'she is', text)
    text = re.sub(r'\bthat\'s\b', 'that is', text)
    text = re.sub(r'\bthere\'s\b', 'there is', text)

    text = re.sub(r'\bit\'re\b', 'it are', text)
    text = re.sub(r'\bhe\'re\b', 'he are', text)
    text = re.sub(r'\bshe\'re\b', 'she are', text)
    text = re.sub(r'\bthat\'re\b', 'that are', text)
    text = re.sub(r'\bthere\'re\b', 'there are', text)

    text = re.sub(r'\bnot\'ve\b', 'not have', text)
    text = re.sub(r'\bit\'ll\b', 'it will', text)

    text = re.sub(r'\bi\'d\b', 'i would', text)
    text = re.sub(r'\bwe\'d\b', 'we would', text)
    text = re.sub(r'\byou\'d\b', 'you would', text)
    text = re.sub(r'\bhe\'d\b', 'he would', text)
    text = re.sub(r'\bshe\'d\b', 'she would', text)
    text = re.sub(r'\bit\'d\b', 'it would', text)
    text = re.sub(r'\bthey\'d\b', 'they would', text)

    text = re.sub(r'\bwasn\'t\b', 'was not', text)

    text = re.sub(r'\bhow\'d\b', 'how would', text)
    text = re.sub(r'\bwhat\'d\b', 'what would', text)
    text = re.sub(r'\bwho\'d\b', 'who would', text)

    text = re.sub(r'\bmrs.\b', 'mister', text)
    text = re.sub(r'\bmrs\b', 'mister', text)


    text = re.sub(r'\bms\b', 'miss', text)
    text = re.sub(r'\bms.\b', 'miss', text)
    return text

def fix_common_mistakes():
    text = re.sub(r"\be g\b", " exammple ", text)
    text = re.sub(r"\be.g.\b", " exammple ", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"e-commerce", "ecommerce", text)

    dict = {"absence": ["abcense", "absance"],
            "acceptable ": ["acceptible"],
            "affect": ["effect"],
            "achieve": ["acheive"],
            "address": ["adress"]
            } # TODO: fill more from here https://en.wikipedia.org/wiki/Commonly_misspelled_English_words

    for key, value in dict.items():
        for value_item in value:
            text = re.sub(value_item, key, text)
    return text

def unstack(text):
    # TODO: fill more
    dict = {r"\bkg\b": "kilogram",
            r"\bkgs\b": "kilograms",
            r"\busa\b ": "America",
            r"\betc\b": "et cetera",
            r"\be.g.\b": "for example"
            }

    for key, value in dict.items():
        text = re.sub(key, value, text)

    return text

def remove_punctuiation(text):
    # Remove punctuation
    # REMOVE COMMENT IF TEXTACY IS INSTALLED
    # text = textacy.preprocess.preprocess_text(text, no_punct=True)
    return text

def remove_white_space(text):
    return re.sub(r"\s+", " ", text)
#

##################################
# TOKENIZATION
##################################
# word_tokenize("jaksldjkalsd ! aksdasd!")
dict = {"absence" :["abcense", "absance"],
        "acceptable " : ["acceptible"],
        }

for key, value in dict.items():
    for value_item in value:
        text = re.sub(r"" + value_item, key, text)

print(text)

dict = {r"\bto\b": "tooooooooooo",
        r"\bkgs\b": "kilograms",
        r"\busa\b ": "America",

        }

for key, value in dict.items():
    text = re.sub(key, value, text)
    print(key, value)
# text = re.sub(r"\bto\b", "asdasd", text)
print(text)

text = remove_punctuiation(text)
print(text)

# TODO: add stemmer at the end
from nltk.stem import PorterStemmer
print("-------------------------------------------")
ps = PorterStemmer()
example = ["python","pythoner","pythoning","pythoned","pythonly"]
st = ""
for w in example:
    print(ps.stem(w))
    st = ps.stem(w)

print(ps.stem(st))
print(type(st))

from nltk.tokenize import word_tokenize
wt = word_tokenize(text)
print(ps.stem(text))
print(set(wt))
for el in set(wt):
    print(ps.stem(el))


