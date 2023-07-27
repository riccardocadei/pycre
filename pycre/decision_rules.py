import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV
from sklearn.utils import resample

from tree import DecisionTree


def generate_rules(X, ite, n_trees=1, max_depth=3, node_size=10, max_rules=50, decimal=2, criterion="het", subsample=0.7):
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
        X_ = X.sample(frac=subsample)
        ite_ = ite[X_.index]
        # decision tree
        model = DecisionTree(max_depth=max_depth, 
                             node_size=node_size, 
                             criterion=criterion)
        model.fit(X_, ite_)
        rules += model.get_rules(decimal)

    # top rules selection
    rules = sorted(list(pd.Series(rules).value_counts()[:max_rules].index))
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
                        subsample = 0.7,
                        alphas = [0.1, 1.0, 10.0]):
    """
    Select rules with stability selection.

    Input:
        R: pd.DataFrame Rules Matrix (N x M)
        ite: pd.Series with ITE estimates (N)
        t_ss: threshold for stability selection
        alpha: list of alpha values for LassoCV
        B: number of bootstrap samples

    
    Output:
        rules: list of selected rules
    """

    M = R.shape[1]
    R = R.div(R.columns.str.count("&")+1, axis=1)
    ite = ite-np.mean(ite)
    stability_scores = np.zeros(M)
    for _ in range(B):
        X, y = resample(R, ite, replace=False, n_samples=int(len(R) * subsample))
        lasso = LassoCV(alphas=alphas, cv=5).fit(X, y)
        non_zero_indices = np.where(lasso.coef_ != 0)[0]
        stability_scores[non_zero_indices] += 1
    stability_scores /= B

    rules = list(R.columns[stability_scores >= t_ss])   
    return rules