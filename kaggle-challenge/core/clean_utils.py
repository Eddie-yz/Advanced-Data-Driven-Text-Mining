import re
import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words('english')

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_punc(text):
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return text.translate(table)

def remove_digits(text): 
    pattern = '[0-9]'
    text = re.sub(pattern, '', text)
    return text

def html_unescape(text):
    return html.unescape(text)

def reduce_length(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

#tokenize sentence and correct the spelling
def token_n_spellcheck(text):
    words = word_tokenize(text)
    reduced_text  = [reduce_length(word) for word in words]
    stemmer = SnowballStemmer("english")
    stem_text = [stemmer.stem(word) for word in reduced_text if word not in stop_words]

    return [word for word in stem_text if len(word) >=3]

# the pipeline function for text cleaning
def text_clean(text):
    text = text.lower()
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_digits(text)
    text = remove_punc(text)
    words = token_n_spellcheck(text)
    return words