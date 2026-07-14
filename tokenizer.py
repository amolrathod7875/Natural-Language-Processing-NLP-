import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
text = "Hey there my name is Amol i am a man "
tokens = word_tokenize(text)

stop_words = stopwords.words('english')

filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print(filtered_tokens)