import itertools
from scipy.special import comb
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import GeneralModel as gm


"""
    Given a set of categorical and continuous features, the function will iterate
    over the combinations of features and present the results in a dataframe
    
    Columns in dataframe: features (list), fscore, auc, feature importance (if applicable) 
    
    input:
        - df: original dataframe
        - min_feats: minimum subset size taken from feature list
        - sort: column by which to sort dataframe (by AUC or fscore)
        - algorithm: type of algorithm
        - cat_vars: list of categorical variables
        - cont_vars: list of continuous variables
"""
def model_evaluation(df, min_feats=1, sort=None, algorithm='rf', cat_vars=[], cont_vars=[]):
    total_features = cat_vars + cont_vars
    
    # keeps track of results
    results = pd.DataFrame()
    
    if min_feats > len(total_features):
        print('error: minimum subset size exceeds feature set')
        return None
    
    # iterates through all subsets greater than min_feats
    for i in range(len(total_features)-min_feats+1):
        subset_size = i + min_feats
        
        num_combinations = int(comb(len(total_features), subset_size))
        print('# of combinations for subset size {}: {}'.format(subset_size, num_combinations))
        for var_set in tqdm(itertools.combinations(total_features, subset_size)):
            # separates the feature subset into categorical and continuous
            cont_var_set = [var for var in var_set if var in cont_vars]
            cat_var_set = [var for var in var_set if var in cat_vars]
            
            # prepares model for feature subset
            model_df = gm.prepare_df(df, cont_vars=cont_var_set, cat_vars=cat_var_set, print_dims=False)
            fscore, _, auc_score, feat_importance_df = gm.general_model(model_df, algorithm=algorithm, print_metrics=False, tqdm_on=False)
        
            feature_string = ', '.join(list(var_set))
            num_feats = subset_size
            
            # handles feature importance field
            ordered_feat_importance = feat_importance_df.sort_values('Weight', ascending=False)
            ordered_weights = ordered_feat_importance['Weight']
            ordered_feats = ordered_feat_importance['Feature']
            
            import_array = []
            for i in range(len(ordered_feat_importance.index)):
                import_array.append('{}: {}'.format(ordered_feats[i], ordered_weights[i]))
            import_string = ', '.join(import_array)
            
            # adds result of model to dataframe
            results = results.append(pd.DataFrame({'features': feature_string, 'fscore': fscore, 
                                                   'auc': auc_score, 'num_features': num_feats, 'feat_importance': import_string}, index=[0]))
    
    # sorts dataframe in descending order of desired metric
    if sort is not None : results = results.sort_values(sort, ascending=False)
        
    # columns in right order
    col_order = [sort, 'features'] + [col for col in results.columns if col not in [sort, 'features']]
    results = results[col_order]
    
    return results