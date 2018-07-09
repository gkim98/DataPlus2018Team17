"""
    Text classification with stratified k-fold cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import GridSearchCV
from TextClassification import GRID_DEFAULT
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix


"""
    Performs stratified k-fold cross-validation on text classification model

    input: 
        - df: dataframe with transcripts
        - grid_search: whether to perform grid search
        - hyp_params: hyperparameters for grid search
        - folds: number of folds
        - iterations: number of times performing cross-validation
"""
def strat_kfold_text(df, grid_search=True, hyp_params=GRID_DEFAULT, folds=3, iterations=1, print_results=True):
    avg_pos_precision=0
    avg_pos_recall=0
    avg_neg_precision=0
    avg_neg_recall=0

    X = df['Convo_1'].values
    y = df['txgot_binary'].values

    # keeps track of decision 
    dec_values = np.zeros(X.size)

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=iterations)
    for train_index, test_index in tqdm(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        (tn, fp, fn, tp), decision_function = train_text(X_train, y_train, X_test, y_test, grid_search, hyp_params)

        # aggregating decision function values
        dec_result = decision_function(X_test)
        for i in range(len(test_index)):
            dec_values[test_index[i]] += dec_result[i]

        # aggregrating evaluation metrics
        avg_pos_precision += tp / (tp + fp)
        avg_pos_recall += tp / (tp + fn)
        avg_neg_precision += tn / (tn + fn)
        avg_neg_recall += tn / (tn + fp)

    # takes averages
    avg_pos_precision /= (iterations * folds)
    avg_pos_recall /= (iterations * folds)
    avg_neg_precision /= (iterations * folds)
    avg_neg_recall /= (iterations * folds)
    dec_values /= iterations

    metrics = {
        "positive precision": avg_pos_precision,
        "positive recall": avg_pos_recall,
        "negative precision": avg_neg_precision,
        "negative recall": avg_neg_recall
    }

    if print_results:
        print('AVG METRICS:\n')
        print('Surveillance Class Precision: {}\n'.format(round(avg_pos_precision, 3)))
        print('Surveillance Class Recall: {}\n'.format(round(avg_pos_recall, 3)))
        print('Treatment Class Precision: {}\n'.format(round(avg_neg_precision, 3)))
        print('Treatment Class Recall: {}\n'.format(round(avg_neg_recall, 3)))

    return dec_values, metrics


"""
    One iteration of text classification model

"""
def train_text(X_train, y_train, X_test, y_test, grid_search, hyp_params):
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))
    ])

    if grid_search:
        model = GridSearchCV(text_clf, hyp_params, n_jobs=-1)
        model = model.fit(X_train, y_train)
    else:
        model = text_clf.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = confusion_matrix(y_test, predictions).ravel()

    # returns decision function
    decision_function = model.decision_function

    print(metrics)
    return metrics, decision_function

