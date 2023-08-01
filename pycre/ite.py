from sklearn.ensemble import GradientBoostingClassifier

from econml.dml import CausalForestDML
from econml.metalearners import TLearner, SLearner, XLearner
from econml.dr import DRLearner

import pandas as pd

def estimate_ite(X, y, z,
                 method = 'slearner',
                 learner_y = GradientBoostingClassifier(),
                 learner_ps = GradientBoostingClassifier()):
    """
    Estimate Pseudo-Outcome (ITE)

    Parameters
    ----------
    X: pd.DataFrame 
        Covariates Matrix (N x P)
    y: pd.Series
        Outcome Vector (N)
    z: pd.Series
        Treatment Vector (N)
    method: {'tlearner', 'slearner', 'xlearner', 'aipw', 'drlearner', 
        'causalforest'}, default="slearner"
        Pseudo-Outcome (ITE) estimator.
    learner_y : sklearn learner, default=GradientBoostingClassifier()
            model for outcome estimation, 
    learner_ps : sklearn learner, default=GradientBoostingClassifier()
            model for propensity score estimation
    
    Returns
    -------
    pd.Series 
        ITE estimates (N)
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
        model = AIPW(model_propensity = learner_ps,
                     model_outcome = learner_y)
    elif method == 'drlearner':
        model = DRLearner(model_propensity = learner_ps,
                          model_regression = learner_y,
                          model_final = learner_y)
    elif method == 'causalforest':
        model = CausalForestDML(discrete_treatment = True,
                                model_t = learner_ps, 
                                model_y = learner_y)
    else:
        raise ValueError(f"""{method} method for ITE estimation not 
                         implemented""")
    model.fit(Y=y, T=z, X=X)
    ite = pd.Series(model.effect(X), index=X.index)
    return ite

class AIPW:
    def __init__(self, model_propensity, model_outcome):
        self.model_propensity = model_propensity
        self.model_outcome = model_outcome

    def fit(self, Y, T, X):
        self.model_propensity.fit(X = X, y = T)
        self.model_outcome.fit(X = X.assign(z=T), y = Y)
        mu0 = self.model_outcome.predict(X.assign(z=0))
        mu1 = self.model_outcome.predict(X.assign(z=1))
        ps = self.model_propensity.predict_proba(X)[:, 1]
        ite = mu1-mu0 + T * (Y-mu1) / (ps) - (1-T) * (Y-mu0) / (1-ps)    
        self.ite = pd.Series(ite, index=X.index)

    def effect(self, X):
        return pd.Series(self.ite, index=X.index)
