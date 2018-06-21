"""
    This will compile ideas from other model scripts into a general model that can take any features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# silences package warnings
import warnings
warnings.filterwarnings('ignore')


GRID_DEFAULT = {
    'C': [0.1,1, 10, 100, 1000], 
    'gamma': [1,0.1,0.01,0.001,0.0001], 
    'kernel': ['rbf']
} 


"""
    Trains and evaluates model based on given features
    
    input:
        - df: data for the model
        - algorithm: type of algorithm for training model (svm, rf, lr)
        - pred_var: name of the target variable
        - hyp_params: parameters for grid search
        - folds: folds for cross-validation
        - iterations: repititions of kfold cross-validation
        - print_results: whether to print results
        - tqdm_on: whether to use tqdm progress bar
    output:
        - F-score of model
        - dictionary of metrics
"""
def general_model(df, algorithm='svm', pred_var='txgot_binary', folds=10, iterations=3, print_results=True, tqdm_on=True):
    avg_pos_prec = 0
    avg_pos_rec = 0
    avg_neg_prec = 0
    avg_neg_rec = 0

    # maps algorithm parameter to algorithm function
    alg_map = {
        'svm': train_svm_model,
        'rf': train_rf_model,
        'lr': train_lr_model
    }

    # treats every non-target variable as a feature
    feat_vars = [var for var in list(df.columns) if var != pred_var]

    X = df[feat_vars].as_matrix()
    y = df[pred_var].as_matrix()

    # keeps track of feature importance/coefficients
    feat_info = np.zeros((1, len(df[feat_vars].columns)))

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=iterations)
    # toggles tqdm
    splits = tqdm(rskf.split(X, y)) if tqdm_on else rskf.split(X, y)

    for train_index, test_index in splits:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        (tn, fp, fn, tp), weights = alg_map[algorithm](X_train, y_train, X_test, y_test)

        if weights.all():
            feat_info += weights

        avg_pos_prec += tp / (tp + fp)
        avg_pos_rec += tp / (tp + fn)
        avg_neg_prec += tn / (tn + fn)
        avg_neg_rec += tn / (tn + fp)

    # calculates average precision and recall
    avg_pos_prec /= (folds * iterations)
    avg_pos_rec /= (folds * iterations)
    avg_neg_prec /= (folds * iterations)
    avg_neg_rec /= (folds * iterations)

    # gets average of weights and displays
    if weights.all():
        if algorithm == 'lr':
            weights = weights[0]
        feat_info /= (folds * iterations)
        feat_importance_df = pd.DataFrame({"Feature": df[feat_vars].columns, "Weight": np.transpose(weights)})
        print(feat_importance_df)
        print()


    if print_results:
        print('Average Metrics:')
        print('Positive Class Precision: {}'.format(round(avg_pos_prec, 3)))
        print('Positive Class Recall: {}'.format(round(avg_pos_rec, 3)))
        print('Negative Class Precision: {}'.format(round(avg_neg_prec, 3)))
        print('Negative Class Recall: {}'.format(round(avg_neg_rec, 3)))

    metrics = {
        'positive precision': avg_pos_prec,
        'positive recall': avg_pos_rec,
        'negative precision': avg_neg_prec,
        'negative recall': avg_neg_rec 
    }
    fscore = (2 * avg_pos_prec * avg_pos_rec) / (avg_pos_prec + avg_pos_rec)

    return fscore, metrics



"""
    Model training for different algorithms
"""
# trains one iteration of svm model
def train_svm_model(X_train, y_train, X_test, y_test, *args):
    # trains model
    model = GridSearchCV(SVC(), GRID_DEFAULT, refit=True)
    model.fit(X_train, y_train)

    # make predictions and evaluate
    predictions = model.predict(X_test)
    metrics = confusion_matrix(y_test, predictions).ravel()
    return metrics, None

# trains one iteration of random forest model
def train_rf_model(X_train, y_train, X_test, y_test, *args):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # displays feature importances
    feat_importances = model.feature_importances_

    predictions = model.predict(X_test)
    metrics = confusion_matrix(y_test, predictions).ravel()
    return metrics, feat_importances


# trains one iteration of 
def train_lr_model(X_train, y_train, X_test, y_test, *args):
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # displays coefficients of the features
    coefficients = np.asarray(model.coef_)

    predictions = model.predict(X_test)
    metrics = confusion_matrix(y_test, predictions).ravel()
    return metrics, coefficients
    


"""
    Prepares a dataframe for general_model

    input:
        - df: untouched dataframe
        - cont_vars: continuous variables as list
        - cat_vars: categorical variables as list
        - target_var: target variable
        - print_dims: whether to print out # of examples
    output:
        - dataframe ready to feed through general_model

    ADD SOMETHING TO CONTROL DROPPING OF NA VALUES
"""
def prepare_df(df, cont_vars=['age'], cat_vars=['gleason'], target_var='txgot_binary', print_dims=True):
    total_vars = cont_vars + cat_vars + [target_var]
    model_df = df[total_vars]
    cleaned_df = model_df.dropna(subset=total_vars)

    # turns categorical variables into dummy variables
    for var in cat_vars:
        temp_dummy = pd.get_dummies(cleaned_df[var], drop_first=True)
        cleaned_df = pd.concat([cleaned_df.drop([var], axis=1), temp_dummy], axis=1)

    # normalize the data
    for var in cont_vars:
        cleaned_df[var] = preprocessing.scale(cleaned_df[var])

    if print_dims:
        print('# of Data Points: {}'.format(len(cleaned_df.index)))

    return cleaned_df