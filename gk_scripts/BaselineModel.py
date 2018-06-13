"""
    Baseline predictive model, using only Gleason score and age as features
    (only SVM)
"""

import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score


GRID_DEFAULT = {
    'C': [0.1,1, 10, 100, 1000], 
    'gamma': [1,0.1,0.01,0.001,0.0001], 
    'kernel': ['rbf']
} 


"""
    Trains model and outputs results

    input: 
        - df: dataframe cut down for the model
        - grid_search: whether to perform grid search
        - hyp_params: hyperparameters to test for grid search
        - iterations: number of evaluations of model
        - test_size: percentage of examples in test set
    output: returns model, prints model performance
"""
def baseline_model(df, grid_search=False, hyp_params=GRID_DEFAULT, iterations=1, test_size=0.33):

    # turn Gleason into dummy variable
    gleason_dummy = pd.get_dummies(df['gleason'], drop_first=True)
    model_df = pd.concat([df.drop(['gleason'], axis=1), gleason_dummy], axis=1)

    # keeps track of precision and recall over iterations
    avg_precision = 0
    avg_recall = 0
    # runs model iterations times
    for i in tqdm(range(iterations)):
        prec, rec = train_model(model_df, grid_search, hyp_params, test_size)
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
    Trains a model for a single iteration

    input: 
        - df: model's dataframe
        - grid_search: whether to perform grid search
        - hyp_params: hyperparameters for grid search
        - test_size: fraction of examples in test set
    output: recall and precision for each class 
"""
def train_model(df, grid_search, hyp_params, test_size):
    # get train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('active_surv', axis=1),
        df['active_surv'], test_size=test_size)

    # trains model
    model = SVC()
    if grid_search:
        model = GridSearchCV(SVC(), hyp_params, refit=True)
    model.fit(X_train, y_train)

    # make predictions and evaluate
    predictions = model.predict(X_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    return precision, recall


"""
    Processes the dataframe for use in model

    input: merged dataframe
    output: processed dataframe for predictive model
"""
def model_dataframe(df):
    df = df.copy()
    model_df = df[['gleason', 'age', 'active_surv']]
    clean_df = model_df.dropna()
    return clean_df



