"""
    Performs topic modeling with LDA
"""

import gensim
from gensim import corpora
import pickle
import pandas as pd
import Preprocessing as pre


TOPIC_PIPELINE = [
    pre.make_lowercase,
    pre.remove_parentheses,
    pre.remove_non_alpha,
    pre.remove_stopwords,
    lambda x: pre.remove_stopwords(x, pre.CUSTOM_STOPWORDS),
    pre.lemmatize
]


"""
    Takes corpus and performs LDA topic modeling

    input:
        - corpus: convos as a list of strings
        - num_topics: number of topics
        - passes: passes through algorithm? (more research)
        - num_words_print: number of words to store for each topic
        - print_topics: whether to print topics (True/False)
        - pickle: whether to pickle dictionary (True/False)
        - preprocess: whether preprocessing is needed (True/False)
    output: 
        - returns a list of tuples representing the topics
"""
def topic_model(corpus, num_topics, passes, num_words_print=5, print_topics=False, pickle=False, preprocess=False):
    if preprocess:
        # if corpus is a list of unprocessed strings
        corpus_array = [pre.text_preprocessing(convo, TOPIC_PIPELINE) for convo in corpus]
    else:
        # if corpus is already a list of tokenized word lists (e.g. from pickle)
        corpus_array = corpus

    dictionary = corpora.Dictionary(corpus_array)
    corpus_dict = [dictionary.doc2bow(text) for text in corpus_array]

    # pickles dictionary for later use
    if pickle:
        pickle.dump(corpus, open('corpus.pkl', 'wb'))
        dictionary.save('dictionary.gensim')

    # performs LDA topic modeling
    ldamodel = gensim.models.ldamodel.LdaModel(corpus_dict, num_topics=num_topics, id2word=dictionary, passes=passes)
        
    topics = ldamodel.print_topics(num_words=num_words_print)

    # prints topics
    if print_topics:
        for topic in topics:
            print(topic)

    return topics

    

