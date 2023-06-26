from sklearn.ensemble import GradientBoostingClassifier

from econml.dml import CausalForestDML, LinearDML
from econml.metalearners import TLearner, SLearner, XLearner
from econml.dr import DRLearner

import pandas as pd

def estimate_ite(X, y, z,
                 method = 'slearner',
                 learner_y = GradientBoostingClassifier(),
                 learner_ps = GradientBoostingClassifier()):
    """
    Estimate ITE

    Input
        X: pd.DataFrame with Covariates ('X')
        y: pd.Series with Outcome ('y')
        z: pd.Series with Treatment ('z')
        learner_y: sklearn learner for outcome estimation
        learner_ps: sklearn learner for propensity score estimation
        method: method for ITE estimation
    
    Output
        ite: pd.Series with ITE estimates (N)
    """

    if method == 'tlearner':
        model = TLearner(models = learner_y)
    elif method == 'slearner':
        model = SLearner(overall_model = learner_y)
    elif method == 'xlearner':
        model = XLearner(models = learner_y,
                         propensity_model = learner_ps,
                         cate_models = learner_y)
    elif method == 'aipw':
        model = LinearDML(model_y = learner_y,
                            model_t = learner_ps)
        
    elif method == 'drlearner':
        model = DRLearner(model_propensity = learner_ps,
                          model_regression = learner_y,
                          model_final = learner_y)
    elif method == 'causalforest':
        # TODO: finetune hyperparameters
        model = CausalForestDML(criterion = 'mse', 
                                n_estimators = 500,       
                                min_samples_leaf = 5, 
                                max_depth = 2, 
                                discrete_treatment = True,
                                model_t = learner_ps, 
                                model_y = learner_y)
    else:
        raise ValueError(f'{method} method for ITE estimation not implemented')
    model.fit(Y=y, T=z, X=X)
    ite = pd.Series(model.effect(X), index=X.index)
    
    return ite