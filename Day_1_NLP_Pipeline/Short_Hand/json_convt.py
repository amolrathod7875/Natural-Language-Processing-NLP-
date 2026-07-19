import string 
from textblob import TextBlob

def load_slang_dic(filepath='slang.txt'):
    slang_dic = {}
    
    with open(filepath,'r',encoding='utf-8') as f :
        for line in f:
            line = line.strip()
            if '=' in line:
                abbrev,meaning = line.split('=',1)
                slang_dic[abbrev.strip()] = meaning.strip()
    return slang_dic

def expand_slang(text,slang_dic):
    import re
    for abbrev in sorted(slang_dic.keys(), key=len, reverse=True):
        pattern = r'\b' + re.escape(abbrev) + r'\b'
        text = re.sub(pattern,slang_dic[abbrev], text, flags=re.IGNORECASE)
    return text

slang_from_dic = load_slang_dic()
text = "hey there bbby, call me ASAP !"
expended = expand_slang(text, slang_from_dic)
print(f"Original Text : {text}")
print(f"Slang_Removed : {expended}")

punc = string.punctuation

def remove_punc(semi_final):
    for char in punc:
        semi_final = semi_final.replace(char, '')
    return semi_final

final_text = remove_punc(expended)

print(f"Removed Punctuation : {final_text}")

blog = TextBlob(final_text)
print(f"Correct Order : {blog.correct().string}")



