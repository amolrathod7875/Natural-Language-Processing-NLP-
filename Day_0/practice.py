import nltk 

text = "MMy name is amol. I am from, Modern COllege "
sentence = nltk.sent_tokenize(text)
sentence_ = "hey there UV i like u  "
words = nltk.word_tokenize(sentence_)
print(sentence)
print(words)
