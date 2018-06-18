"""
    Playing around with stratified k-fold cross-validation with same model as BaselineModel
"""

import BaselineModel as bm
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


"""
    Trains model with K-fold cross validation 

    input: 
        - df: model's dataframe
        - folds: how many folds for cross-validation
        - iterations: how many times cross-validation was performed
        - grid_search: whether to perform grid search
        - hyp_params: parameters to check during grid search
        - 
    output: 
        - precision and recall of surveillance class
"""
def strat_kfold_model(df, folds=3, iterations=1, grid_search=False, hyp_params=bm.GRID_DEFAULT, print_results=True):
    avg_pos_precision=0
    avg_pos_recall=0
    avg_neg_precision=0
    avg_neg_recall=0

    X = df[['age', 'gleason']].as_matrix()
    y = df['active_surv'].as_matrix()

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=iterations)
    for train_index, test_index in tqdm(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tn, fp, fn, tp = train_model(X_train, y_train, X_test, y_test, grid_search, hyp_params)
        avg_pos_precision += tp / (tp + fp)
        avg_pos_recall += tp / (tp + fn)
        avg_neg_precision += tn / (tn + fn)
        avg_neg_recall += tn / (tn + fp)

    # calculates average precision and recall
    avg_pos_precision /= (iterations * folds)
    avg_pos_recall /= (iterations * folds)
    avg_neg_precision /= (iterations * folds)
    avg_neg_recall /= (iterations * folds)
    
    if print_results:
        print('AVG METRICS:\n')
        print('Surveillance Class Precision: {}\n'.format(round(avg_pos_precision, 3)))
        print('Surveillance Class Recall: {}\n'.format(round(avg_pos_recall, 3)))
        print('Treatment Class Precision: {}\n'.format(round(avg_neg_precision, 3)))
        print('Treatment Class Recall: {}\n'.format(round(avg_neg_recall, 3)))

    return avg_pos_precision, avg_pos_recall, avg_neg_precision, avg_neg_recall


"""
    Trains one iteration of the model
"""
def train_model(X_train, y_train, X_test, y_test, grid_search, hyp_params):
    # trains model
    model = SVC()
    if grid_search:
        model = GridSearchCV(SVC(), hyp_params, refit=True)
    model.fit(X_train, y_train)

    # make predictions and evaluate
    predictions = model.predict(X_test)
    metrics = confusion_matrix(y_test, predictions).ravel()
    return metrics