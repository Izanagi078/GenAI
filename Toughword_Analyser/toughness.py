import re
from wordfreq import word_frequency
import nltk

# Ensure punkt and stopwords are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def tokenize(text):
    return [w.lower() for w in nltk.word_tokenize(text, preserve_line=True)
            if re.match(r'\w+', w) and w.lower() not in STOPWORDS]

def toughness_score(word):
    freq = word_frequency(word, 'en')
    if freq == 0.0:
        freq = 1e-9  # Avoid zero division
    length = len(word)
    score = (1 - freq) * length  # Composite metric
    return score

def get_top_tough_words(text, top_n=3):
    words = tokenize(text)
    unique_words = list(set(words))
    scored = [(w, toughness_score(w)) for w in unique_words]
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    return [w for w, _ in top]
