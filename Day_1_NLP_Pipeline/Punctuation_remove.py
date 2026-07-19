import string
from textblob import TextBlob


exclude = string.punctuation


def remove_punctuation(text):
    for char in exclude:
        text = text.replace(char,'')
    return text


unpunc = remove_punctuation("Hello !.What is your nameee ?")
print(unpunc)

blob = TextBlob(unpunc)

#print(blob.correct())