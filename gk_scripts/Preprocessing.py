"""
    Performs basic text preprocessing, such as removal of punctuaction, removal
    of stopwords, and lemmatization

    Methods:
    - lemmatize
    - make_lowercase
    - remove_non_alpha
    - remove_parentheses
    - remove_punct
    - remove_stopwords
    - text_processing (pass in other functions as pipeline)
"""

import nltk
import string
from tqdm import tqdm_notebook as tqdm
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


"""
    Defaults for below functions
"""
defaults = {
    'punct': string.punctuation,
    'stop': set(stopwords.words('english'))
}

CUSTOM_STOPWORDS = ['pt', 'md', 'ok', 'yeah', 'okay', 'um', 'uh']


# helper function for lemmatize
def lemmatize_helper(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

"""
    Lemmatizes

    input: text represented as tokenized list of words
    output: list of lemmatized words
"""
def lemmatize(words):
    lemmatized_words = [lemmatize_helper(word) for word in words]
    return lemmatized_words


"""
    Turns all words into lowercase

    input: text represented as tokenized list of words
    output: list of words without any capitalization
"""
def make_lowercase(words):
    filtered_words = [word.lower() for word in words]
    return filtered_words


"""
    Removes any non-alphabetic terms from list of words

    input: text represented as tokenized list of words
    output: list of words without non-alphabetic terms
"""
def remove_non_alpha(words):
    filtered_words = [word for word in words if word.isalpha()]
    return filtered_words


"""
    Removes transcriber notes, denoted by parantheses
    
    input: text represented as tokenized list of words
    output: list of words without transcriber additions
"""
def remove_parentheses(words):
    non_parenth = []
    
    # keeps track if word is in parentheses
    parenth_debt = 0
    for i in range(len(words)):
        if words[i]=='(' : parenth_debt += 1
        if parenth_debt==0 : non_parenth.append(words[i])
        if words[i]==')' : parenth_debt -= 1
            
    return non_parenth


"""
    Removes unwanted punctuation from text
    
    input: text represented as tokenized list of words, list of unwanted punct
    output: list of words without unwanted punctuation
"""
def remove_punct(words, puncts=defaults['punct']):
    filtered_words = [word for word in words if word not in puncts]
    return filtered_words


"""
    Removes stopwords from text
    
    input: text represented as tokenized list of words, list of stopwords
    output: list of words without stopwords
"""
def remove_stopwords(words, stopwords=defaults['stop']):
    filtered_words = [word for word in words if word not in stopwords]
    return filtered_words


"""
    Runs text through customizable preprocessing

    inputs:
        text - text as a string to preprocess
        steps - list of desired preprocessing functions
    output: preprocessed text as tokenized list of words
"""
def text_preprocessing(text, steps):
    words = nltk.word_tokenize(text)

    # runs steps outlined in function list
    for i in range(len(steps)):
        words = steps[i](words)
    
    return words