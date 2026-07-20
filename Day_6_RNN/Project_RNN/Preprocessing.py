import re
import html
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Datacleaner: 
    @staticmethod
    def remove_html(data):
        if isinstance(data,str):
            data = html.unescape(data)
            data = re.sub(r'<[^>]+>',' ',data)
            data = re.sub(r'\s+',' ',data)
            return data.strip()
        return data 
    
    @staticmethod
    def remove_punctuation(data):
        if isinstance(data,str):
            punctuation = string.punctuation
            return ''.join(char for char in data if char not in punctuation)
        return data
    
    @staticmethod
    def remove_duplicated(df):
        df.drop_duplicates(inplace=True)
        return df
    
    @staticmethod
    def lower_string(data):
        if isinstance(data,str):
            return data.lower()
        return data
    
    @staticmethod
    def tokenize_text(text):
        if isinstance(text,str) and text.lower():
            return word_tokenize(text.lower())
        return []

    @staticmethod 
    def remove_stopwords(data):
        stop_words = set(stopwords.words('english'))
        if isinstance(data,list):
            return [word for word in data if word.lower() not in stop_words]
        return []
    



