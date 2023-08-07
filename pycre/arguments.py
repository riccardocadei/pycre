from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin, RegressorMixin

import pandas as pd
import numpy as np

def check_args(args):
    """
    Check arguments format and dimension
    
    Parameters
    ----------
    args: argparse.Namespace
        parameters
    """
    # Reproducibility
    if args.ratio_dis < 0 or args.ratio_dis > 1:
        raise ValueError("'ratio_dis' parameter must be between 0"
                         "and 1")
    
    # Discovery
    if args.n_trees < 1:
        raise ValueError("'n_trees' parameter must be greater than 0")
    if args.max_depth < 1:
        raise ValueError("'max_depth' parameter must be greater than "
                         "0")
    if args.node_size < 1:
        raise ValueError("'node_size' parameter must be greater than "
                         "0")
    if args.max_rules < 1:
        raise ValueError("'max_rules' parameter must be greater than " 
                         "0")
    if args.decimal < 0:
        raise ValueError("'decimal' parameter must be greater than "
                         "or equal to 0")
    if args.t_ext < 0 or args.t_ext > 1:
        raise ValueError("'t_ext' parameter must be between 0 and "
                         "1")
    if args.t_corr < 0:
        raise ValueError("'t_corr' parameter must be positive")
    if args.t_ss < 0 or args.t_ss > 1:
        raise ValueError("'t_ss' parameter must be between 0 and "
                         "1")
    if args.B < 1:
        raise ValueError("'B' parameter must be greater than 0")
    if args.subsample < 0 or args.subsample > 1:
        raise ValueError("'subsample' parameter must be between 0 "
                         "and 1")
    
    # General
    if not isinstance(args.learner_y, BaseEstimator):
        raise ValueError("'learner_y' parameter must be a sklearn "
                         "estimator")
    if not isinstance(args.learner_ps, BaseEstimator):
        raise ValueError("'learner_ps' parameter must be a sklearn " 
                         "estimator")
    if args.method not in ['tlearner', 'slearner', 'xlearner', 'aipw', 
                           'drlearner', 'causalforest']:
        raise ValueError("'method' parameter doesn't exist or it "
                         "hasn't been implemented yet")

def check_data(X, y, z, learner_y, W=None):
    """
    Check data format and dimension

    Parameters
    ----------
    X: pd.DataFrame 
        Covariates Matrix (N x K)
    y: pd.Series or np.ndarray
        Outcome (N)
    z: pd.Series or np.ndarray
        Treatment (N)
    learner_y : sklearn learner
        model for outcome estimation
    W: pd.DataFrame, default=None
        Additional Covariates Matrix (N x J)
    """
    # check datatype
    if not isinstance(X, pd.DataFrame):
        raise ValueError("'X' must be a pandas DataFrame")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError("'y' must be a pandas Series")
    if not isinstance(z, (pd.Series, np.ndarray)):
        raise ValueError("'z' must be a pandas Series")
    if W is not None and not isinstance(W, pd.DataFrame):
        raise ValueError("'W' must be a pandas DataFrame")
    
    # check X, y and z have the same length
    if not len(X) == len(y) == len(z):
        raise ValueError("'X', 'y' and 'z' must have the same length")
    if W is not None and not len(X) == len(W):
        raise ValueError("'X' and 'W' must have the same length")
    
    # check learner_y is a classifier or a regressor
    binary_y = len(np.unique(y))==2
    if binary_y and not isinstance(learner_y, ClassifierMixin):
        print("WARNING: 'learner_y' is not a (sklearn) classifier, " 
              "but 'y' is binary")
    if not binary_y and not isinstance(learner_y, RegressorMixin):
        print("WARNING: 'learner_y' is not a (sklearn) regressor, "
              "but 'y' is continuos")
    
        




    