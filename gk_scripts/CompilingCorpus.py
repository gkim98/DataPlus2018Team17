"""
    Compiles corpus into a list of strings

    Methods:
    - compile_convos
"""

import pandas as pd


"""
    Creates an array containing all the conversations

    input: transcript dataframe
    output: array of all the transcripts as strings
"""
def compile_convos(df):
    convo_array = []

    for var in ['Convo_1', 'Convo_2']:
        for text in df[var]:
            if isinstance(text, str):
                convo_array.append(text)

    return convo_array


"""
    Takes a tokenized corpora and untokenizes it

    input: corpus as a list of tokenized lists
    output: corpus as a list of strings
"""
def untokenize(corpus):
    untokenized_corp = [" ".join(text) for text in corpus]
    return untokenized_corp