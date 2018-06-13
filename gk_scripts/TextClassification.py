"""
    Classifies text solely based on Convo 1
"""

import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score
import CompilingCorpus as cc
import Preprocessing as pre
from TopicModeling import TOPIC_PIPELINE


GRID_DEFAULT = {
    'vect__ngram_range': [(1, 1), (1, 2)], 
    'tfidf__use_idf': (True, False), 
    'clf__alpha': (1e-2, 1e-3)
}


"""
    Classifies active surveillance based on first conversation transcript

    input: 
        - df: dataframe processed for model
        - test_size: fraction of data to use as test set
        - iterations: iterations through the model
        - grid_search: whether to perform grid search
        - hyp_params: parameters for grid search
    output: classification model
"""
def classify_text(df, test_size=0.33, iterations=1, grid_search=False, hyp_params=GRID_DEFAULT):

    # keeps track of precision and recall over iterations
    avg_precision = 0
    avg_recall = 0
    # runs model iterations times
    for i in tqdm(range(iterations)):
        prec, rec = train_model(df, grid_search, hyp_params, test_size)
        avg_precision += prec
        avg_recall += rec

    # calculates average precision and recall
    avg_precision /= iterations
    avg_recall /= iterations

    print('AVG METRICS:\n')
    print('Surveillance Class Precision: {}\n'.format(round(avg_precision, 3)))
    print('Surveillance Class Recall: {}\n'.format(round(avg_recall, 3)))

    return avg_precision, avg_recall


"""
    Trains text classification model for a single iteration
"""
def train_model(df, grid_search, hyp_params, test_size):
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))
    ])

    X_train, X_test, y_train, y_test = train_test_split(df['Convo_1'], df['active_surv'], test_size=0.33)

    # train model
    if grid_search:
        model = GridSearchCV(text_clf, hyp_params, n_jobs=-1)
        model = model.fit(X_train, y_train)
    else:
        model = text_clf.fit(X_train, y_train)

    # make predictions and evaluate
    predictions = model.predict(X_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    return precision, recall


"""
    Creates dataframe used in text classification model

    input: merged dataframe, transcript dataframe
    output: dataframe for text classification model
"""
def text_df(merged_df, trans_df):
    merged_df = merged_df.copy()
    trans_df = trans_df.copy()

    # formats transcript data
    trans_sub = trans_df[['patient_id', 'Convo_1']]
    trans_sub = trans_sub.dropna(subset=['Convo_1'])
    trans_sub = trans_sub.rename(index=str, columns={'patient_id': 'ID'})

    # formats merged data
    merged_sub = merged_df[['ID', 'active_surv']]

    # merges and takes out unusable examples
    result = pd.merge(trans_sub, merged_sub, on='ID')
    result = result.dropna(subset=['active_surv'])

    return result


"""
    Processes the model's dataframe's conversations

    input: 
        - merged_df: merged dataframe
        - trans_df: transcript dataframe
        - pipeline: steps for text preprocessing
    output: model dataframe with process conversations 
"""
def text_clean_df(merged_df, trans_df, pipeline=TOPIC_PIPELINE):
    df1 = text_df(merged_df, trans_df)
    col_processed = [pre.text_preprocessing(text, pipeline) for text in tqdm(df1['Convo_1'])]
    df1['Convo_1'] = cc.untokenize(col_processed)
    return df1
