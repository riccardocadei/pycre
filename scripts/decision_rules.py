from sklearn.tree import _tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
from sklearn.utils import resample

import numpy as np
import pandas as pd

def generate_rules(X, ite, n_trees=1, max_depth=3, decimal=2):
    """
    Generate decision rules from a set of trees.

    Input:
        X: pd.DataFrame of Covariates
        ite: pd.Series of ITE estiamtes
        n_trees: number of trees to generate
        max_depth: maximum depth of the trees
        decimal: number of digits to round the rules' thresholds

    Output:
        rules: list of candidate decision rules
    """
    
    ite = ite - np.mean(ite)
    rules = []
    for _ in range(n_trees):
        # bootstrap
        X_ = X.sample(frac=0.5)
        ite_ = ite[X_.index]
        # decision tree
        model = DecisionTreeRegressor(max_depth = max_depth)
        model.fit(X_, ite_)
        # visualize
        # print(tree.export_text(model))
        rules += get_rules(model, X_.columns, decimal)
        # discard doubles rules
        rules = sorted(set(rules))
    return rules


def get_rules(tree, feature_names, decimal=2, min_cases=0):
    """
    Get rules from a decision tree.

    Input:
        tree: DecisionTreeRegressor
        feature_names: list of feature names
        decimal: number of digits to round the rules' thresholds
        min_cases: minimum number of cases to consider a rule
    
    Output:
        rules: list of candidate decision rules
    """

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"(X['{name}']<={np.round(threshold, decimal)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"(X['{name}']>{np.round(threshold, decimal)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [tree_.n_node_samples[node]]
            paths += [path]
            
    recurse(0, path, paths)
    
    rules = []
    for path in paths:

        # TO DO: discard too specific rules
        if path[-1]<min_cases: break
        path = path[:-1]

        # clean path
        path.sort()
        for feature_name in feature_names:
            path_i = [node for node in path if feature_name in node]
            if len(path_i)>1:
                path_i_magg = [node for node in path_i if ">" in node]
                if path_i_magg is not None: 
                    if len(path_i_magg)>1:
                        for redundant in path_i_magg[:-1]: 
                            path.remove(redundant)
                path_i_min = [node for node in path_i if "<" in node]
                if path_i_min is not None:
                    if len(path_i_min)>1:
                        for redundant in path_i_min[1:]: 
                            path.remove(redundant)
        rule = ""  
        for p in path:
            if rule != "":
                rule += " & "
            rule += p
        rules += [rule]
        
    return rules


def get_rules_matrix(rules, X):
    """
    Get rules matrix from a list of rules.
    
    Input:
        rules: list of candidate decision rules
        X: pd.DataFrame of Covariates
        
    Output:
        R: pd.DataFrame Rules Matrix (N x M)
    """

    R = {}
    for rule in rules:
        R[rule] = eval(rule).astype(int)
    return pd.DataFrame(R)


def rules_filtering(R, t_ext=0.02, t_corr=0.5):
    """
    Filter rules extreme and correlated rules.
    
    Input:
        R: pd.DataFrame Rules Matrix (N x M)
        t_ext: threshold to discard too generic or too specific (extreme)
        t_corr: threshold to discard too correlated rules
    
    Output:
        R: pd.DataFrame Rules Matrix (N x M)
    """
    
    # disard extreme rules
    generic_rules = R.describe().loc["mean"]>1-t_ext
    rare_rules = R.describe().loc["mean"]<t_ext
    R = R.loc[:,~(generic_rules | rare_rules)]
    if R.shape[1]==0: 
        raise ValueError("No candidates rules left after `extreme rules filtering`. Reduce `t_ext`.")
    
    # discard correlated rules
    corr = R.corr().abs()
    corr_rules = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if corr.iloc[i, j] >= t_corr:
                corr_rules.add(corr.columns[i])
    R = R.drop(columns=corr_rules)
    if R.shape[1]==0:
        raise ValueError("No candidates rules left after `correlated rules filtering`. Increase `t_corr`.")

    return R

def stability_selection(R, ite, 
                        t_ss = 0.6, 
                        B = 50, 
                        alphas = [0.1, 0.5, 1.0],
                        folds = 5):
    """
    Select rules with stability selection.

    Input:
        R: pd.DataFrame Rules Matrix (N x M)
        ite: pd.Series with ITE estimates (N)
        t_ss: threshold for stability selection
        alpha: list of alpha values for LassoCV
        B: number of bootstrap samples
        folds: number of folds for cross validation in LassoCV
    
    Output:
        rules: list of selected rules
    """

    M = R.shape[1]
    ite = ite-np.mean(ite)
    stability_scores = np.zeros(M)
    for _ in range(B):
        X, y = resample(R, ite, replace=False, n_samples=int(len(R) * 0.7))
        lasso = LassoCV(alphas=alphas, cv=folds).fit(X, y)
        non_zero_indices = np.where(lasso.coef_ != 0)[0]
        stability_scores[non_zero_indices] += 1
    stability_scores /= B

    rules = list(R.columns[stability_scores >= t_ss])
    if len(rules)==0:
        raise ValueError(f"No HTE discovered with stability selection threshold `t_ss`={t_ss} (no candidate rules selected`).")
    
    return rules