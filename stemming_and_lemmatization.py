import nltk

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

tokens = ['running','bats','organization','reading']

stemmed = [stemmer.stem(word) for word in tokens]
lemmatizered = [lemmatizer.lemmatize(word) for word in tokens]

print(f"stemming : {stemmed}")
print(f"\nLemmitising : {lemmatizered}")