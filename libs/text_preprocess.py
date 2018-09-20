import re
import time

import pandas as pd
import numpy as np

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk import pos_tag

from nltk.corpus import wordnet


eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()
tokenizer=ToktokTokenizer()


repl = {
        "&lt;3": " good ",
        ":d": " good ",
        ":dd": " good ",
        ":p": " good ",
        "8)": " good ",
        ":-)": " good ",
        ":)": " good ",
        ";)": " good ",
        "(-:": " good ",
        "(:": " good ",
        "yay!": " good ",
        "yay": " good ",
        "yaay": " good ",
        "yaaay": " good ",
        "yaaaay": " good ",
        "yaaaaay": " good ",
        ":/": " bad ",
        ":&gt;": " sad ",
        ":')": " sad ",
        ":-(": " bad ",
        ":(": " bad ",
        ":s": " bad ",
        ":-s": " bad ",
        "&lt;3": " heart ",
        ":d": " smile ",
        ":p": " smile ",
        ":dd": " smile ",
        "8)": " smile ",
        ":-)": " smile ",
        ":)": " smile ",
        ";)": " smile ",
        "(-:": " smile ",
        "(:": " smile ",
        ":/": " worry ",
        ":&gt;": " angry ",
        ":')": " sad ",
        ":-(": " sad ",
        ":(": " sad ",
        ":s": " sad ",
        ":-s": " sad ",
        r"\br\b": "are",
        r"\bu\b": "you",
        r"\bhaha\b": "ha",
        r"\bhahaha\b": "ha",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",
        r"\bcan't\b": "can not",
        r"\bcannot\b": "can not",
        r"\bi'm\b": "i am",
        "m": "am",
        "r": "are",
        "u": "you",
        "haha": "ha",
        "hahaha": "ha",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "can not",
        "cannot": "can not",
        "i'm": "i am",
        "m": "am",
        "i'll" : "i will",
        "its" : "it is",
        "it's" : "it is",
        "'s" : " is",
        "that's" : "that is",
        "weren't" : "were not",
    }

repl_keys = list(repl.keys())




def fix_bad_wording(text):
    arr = str(text).split()
    clean_doc = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in repl_keys:
            # print("inn")
            j = repl[j]
        clean_doc += j + " "
    clean_doc = re.sub('[^a-zA-Z ?!]+', '', str(clean_doc).lower())
    return clean_doc
    
    
#to seperate sentenses into words
def clean_tokenize(text):
    """
    Function to build tokenized texts from input comment
    """
    return simple_preprocess(text, deacc=True, min_len=3)


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
def stop_and_lemmatize(word_list):
    """
    Function to clean the pre-processed word lists 
    
    Following transformations will be done
    1) Stop words removal from the nltk stopword list
    2) Lemmatization (Converting word to its root form : babies --> baby ; children --> child)
    """

    #remove stop words
    clean_words = [w for w in word_list if not w in eng_stopwords]
    
    #Lemmatize
    #lm_words = [lem.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tag(clean_words)]
    lm_words = [] 
    for word in clean_words:
        for pos in ['n', 'v', 'r', 'a', 's']:
            lm = lem.lemmatize(word, pos)
            if lm != word:
                word = lm
                break
        lm_words.append(word)
        
    return(lm_words)

def clean_text(text):
    start_time=time.time()
    text = fix_bad_wording(text)
    print("Fix bad wording: ",time.time()-start_time,"s")
    
    start_time=time.time()
    text = clean_tokenize(text)
    print("Tokenize: ",time.time()-start_time,"s")
    
    start_time=time.time()
    text = stop_and_lemmatize(text)
    print("Remove stopwords and Lemmatize: ",time.time()-start_time,"s")
    print()
    
    return ' '.join(text)


