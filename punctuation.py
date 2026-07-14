import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#print(string.punctuation)

text = "what is the time now ?"
print(text)
token = word_tokenize(text)
print(token)
stop_words = stopwords.words('english')
print(stop_words)
filtered_token = [word for word in token if word.lower() not in stop_words]
print(filtered_token)
clean_word = [word for word in filtered_token if word not in string.punctuation]
print("\n",clean_word)